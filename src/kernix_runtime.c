/* KERNIX 스레드 풀: 작업자를 처음 한 번 만들고 재사용한다 (include/kernix.h의 kernix_pool_run).
 *
 * 작업은 n개 조각으로 나눠 받는다. 호출 스레드와 작업자가 원자적 카운터로 남은 조각을 하나씩 가져가므로,
 * 성능 코어나 한가한 코어가 더 많은 조각을 처리한다. 고정 8등분이면 가장 느린 코어(효율 코어, 부하 걸린 코어)를
 * 모두가 기다려야 한다. */
#include "kernix.h"

#include <stdatomic.h>

static struct {
    pthread_t threads[KERNIX_POOL_WORKERS];
    pthread_mutex_t mutex;     /* 아래 상태와 조건 변수 보호 */
    pthread_cond_t work_cond;  /* 새 작업 또는 종료 */
    pthread_cond_t done_cond;  /* 이번 작업의 작업자가 모두 끝남 */
    size_t generation;
    int active;                /* 이번 작업에서 아직 끝나지 않은 작업자 수 */
    int stopping;
    int initialized;
    int failed;
    /* 이번 작업. generation을 올리기 전에 쓰고, active가 0이 될 때까지 바꾸지 않는다 */
    kernix_chunk_fn fn;
    void* ctx;
    size_t n;
    atomic_size_t next;
} pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .work_cond = PTHREAD_COND_INITIALIZER,
    .done_cond = PTHREAD_COND_INITIALIZER,
};

/* 한 번에 작업 하나만: 여러 스레드가 동시에 부르면 차례로 실행 (작업 상태가 하나뿐이라) */
static pthread_mutex_t run_mutex = PTHREAD_MUTEX_INITIALIZER;

static void drain(void)
{
    size_t c;
    while ((c = atomic_fetch_add_explicit(&pool.next, 1, memory_order_relaxed)) < pool.n) {
        pool.fn(pool.ctx, c);
    }
}

static void* worker(void* arg)
{
    (void)arg;
/* QoS API는 _POSIX_C_SOURCE 등 엄격 모드에서 숨겨짐: 힌트일 뿐이라 생략 */
#if defined(__APPLE__) && __DARWIN_C_LEVEL >= __DARWIN_C_FULL
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    size_t seen = 0;
    pthread_mutex_lock(&pool.mutex);
    for (;;) {
        while (!pool.stopping && pool.generation == seen) {
            pthread_cond_wait(&pool.work_cond, &pool.mutex);
        }
        if (pool.stopping) {
            pthread_mutex_unlock(&pool.mutex);
            return NULL;
        }
        seen = pool.generation;
        pthread_mutex_unlock(&pool.mutex);

        drain();

        pthread_mutex_lock(&pool.mutex);
        if (--pool.active == 0) {
            pthread_cond_signal(&pool.done_cond);
        }
    }
}

static void stop_workers(size_t count)
{
    pthread_mutex_lock(&pool.mutex);
    pool.stopping = 1;
    pthread_cond_broadcast(&pool.work_cond);
    pthread_mutex_unlock(&pool.mutex);
    for (size_t i = 0; i < count; i++) {
        pthread_join(pool.threads[i], NULL);
    }
    pool.stopping = 0;
}

static int init_locked(void)
{
    if (pool.initialized) return 0;
    if (pool.failed) return -1;
    for (size_t i = 0; i < KERNIX_POOL_WORKERS; i++) {
        if (pthread_create(&pool.threads[i], NULL, worker, NULL) != 0) {
            stop_workers(i);
            pool.failed = 1;
            fprintf(stderr, "스레드 풀 생성 실패\n");
            return -1;
        }
    }
    pool.initialized = 1;
    return 0;
}

int kernix_thread_pool_init(void)
{
    pthread_mutex_lock(&run_mutex);
    int rc = init_locked();
    pthread_mutex_unlock(&run_mutex);
    return rc;
}

int kernix_pool_run(size_t n, kernix_chunk_fn fn, void* ctx)
{
    if (fn == NULL) return -1;
    if (n <= 1) { /* 작업자를 깨우는 비용(수 µs)만 듦 */
        if (n == 1) fn(ctx, 0);
        return 0;
    }
    pthread_mutex_lock(&run_mutex);
    if (init_locked() != 0) {
        pthread_mutex_unlock(&run_mutex);
        return -1;
    }
    pthread_mutex_lock(&pool.mutex);
    pool.fn = fn;
    pool.ctx = ctx;
    pool.n = n;
    atomic_store_explicit(&pool.next, 0, memory_order_relaxed);
    pool.active = KERNIX_POOL_WORKERS;
    pool.generation++;
    pthread_cond_broadcast(&pool.work_cond);
    pthread_mutex_unlock(&pool.mutex);

    drain(); /* 호출 스레드도 조각을 처리 */

    pthread_mutex_lock(&pool.mutex);
    while (pool.active > 0) { /* 모든 작업자가 fn/ctx를 다 쓴 뒤에야 반환 (ctx는 호출자 스택일 수 있음) */
        pthread_cond_wait(&pool.done_cond, &pool.mutex);
    }
    pthread_mutex_unlock(&pool.mutex);
    pthread_mutex_unlock(&run_mutex);
    return 0;
}

void kernix_thread_pool_shutdown(void)
{
    pthread_mutex_lock(&run_mutex);
    if (pool.initialized) {
        stop_workers(KERNIX_POOL_WORKERS);
        pool.initialized = 0; /* 다음 kernix_pool_run이 다시 만든다 */
    }
    pthread_mutex_unlock(&run_mutex);
}
