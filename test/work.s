	.build_version macos, 26, 0	sdk_version 26, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_work                           ; -- Begin function work
	.p2align	2
_work:                                  ; @work
	.cfi_startproc
; %bb.0:
	ldr	x8, [x0]
	ldr	x8, [x8, #8]
	ldp	w10, w9, [x0, #8]
                                        ; kill: def $w9 killed $w9 def $x9
	sxtw	x9, w9
	add	w11, w10, #7
	cmp	w11, w9
	b.ge	LBB0_6
; %bb.1:
	sxtw	x13, w10
	sub	x11, x9, #15
	sub	x12, x13, #8
	add	x13, x8, x13, lsl #2
	movi.2d	v0, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v1, #0000000000000000
LBB0_2:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q5, q4, [x13], #32
	fcvtl	v6.2d, v5.2s
	fcvtl	v7.2d, v4.2s
	fcvtl2	v4.2d, v4.4s
	fcvtl2	v5.2d, v5.4s
	fadd.2d	v3, v3, v5
	fadd.2d	v1, v1, v4
	fadd.2d	v2, v2, v7
	fadd.2d	v0, v0, v6
	add	x12, x12, #8
	add	w10, w10, #8
	cmp	x12, x11
	b.lt	LBB0_2
; %bb.3:
	cmp	w10, w9
	b.ge	LBB0_7
LBB0_4:
                                        ; kill: def $w10 killed $w10 killed $x10 def $x10
	sxtw	x11, w10
	sub	x10, x9, x11
	cmp	x10, #4
	b.hs	LBB0_8
; %bb.5:
	mov	x13, x11
	mov.16b	v4, v0
	b	LBB0_18
LBB0_6:
	movi.2d	v0, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v1, #0000000000000000
	cmp	w10, w9
	b.lt	LBB0_4
LBB0_7:
	mov.16b	v4, v0
	b	LBB0_20
LBB0_8:
	cmp	x10, #16
	b.hs	LBB0_10
; %bb.9:
	mov	x12, #0                         ; =0x0
	mov.16b	v4, v0
	b	LBB0_14
LBB0_10:
	and	x12, x10, #0xfffffffffffffff0
	add	x13, x8, x11, lsl #2
	add	x13, x13, #32
	mov	x14, x12
	mov.16b	v4, v0
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldp	q5, q6, [x13, #-32]
	ldp	q7, q16, [x13], #64
	fcvtl2	v17.2d, v5.4s
	mov	d18, v17[1]
	fcvtl	v5.2d, v5.2s
	mov	d19, v5[1]
	fcvtl2	v20.2d, v6.4s
	mov	d21, v20[1]
	fcvtl	v6.2d, v6.2s
	mov	d22, v6[1]
	fcvtl2	v23.2d, v7.4s
	mov	d24, v23[1]
	fcvtl	v7.2d, v7.2s
	mov	d25, v7[1]
	fcvtl2	v26.2d, v16.4s
	mov	d27, v26[1]
	fcvtl	v16.2d, v16.2s
	mov	d28, v16[1]
	fadd	d4, d4, d5
	fadd	d4, d4, d19
	fadd	d4, d4, d17
	fadd	d4, d4, d18
	fadd	d4, d4, d6
	fadd	d4, d4, d22
	fadd	d4, d4, d20
	fadd	d4, d4, d21
	fadd	d4, d4, d7
	fadd	d4, d4, d25
	fadd	d4, d4, d23
	fadd	d4, d4, d24
	fadd	d4, d4, d16
	fadd	d4, d4, d28
	fadd	d4, d4, d26
	fadd	d4, d4, d27
	subs	x14, x14, #16
	b.ne	LBB0_11
; %bb.12:
	cmp	x10, x12
	b.eq	LBB0_20
; %bb.13:
	tst	x10, #0xc
	b.eq	LBB0_17
LBB0_14:
	and	x14, x10, #0xfffffffffffffffc
	add	x13, x14, x11
	lsl	x15, x12, #2
	add	x11, x15, x11, lsl #2
	add	x11, x8, x11
	sub	x12, x12, x14
LBB0_15:                                ; =>This Inner Loop Header: Depth=1
	ldr	q5, [x11], #16
	fcvtl2	v6.2d, v5.4s
	mov	d7, v6[1]
	fcvtl	v5.2d, v5.2s
	mov	d16, v5[1]
	fadd	d4, d4, d5
	fadd	d4, d4, d16
	fadd	d4, d4, d6
	fadd	d4, d4, d7
	adds	x12, x12, #4
	b.ne	LBB0_15
; %bb.16:
	cmp	x10, x14
	b.ne	LBB0_18
	b	LBB0_20
LBB0_17:
	add	x13, x12, x11
LBB0_18:
	add	x8, x8, x13, lsl #2
	sub	x9, x9, x13
LBB0_19:                                ; =>This Inner Loop Header: Depth=1
	ldr	s5, [x8], #4
	fcvt	d5, s5
	fadd	d4, d4, d5
	subs	x9, x9, #1
	b.ne	LBB0_19
LBB0_20:
	movi.2d	v5, #0000000000000000
	fadd	d4, d4, d5
	mov	d0, v0[1]
	fadd	d0, d4, d0
	fadd	d0, d0, d3
	mov	d3, v3[1]
	fadd	d0, d0, d3
	fadd	d0, d0, d2
	mov	d2, v2[1]
	fadd	d0, d0, d2
	fadd	d0, d0, d1
	mov	d1, v1[1]
	fadd	d0, d0, d1
	str	d0, [x0, #16]
	mov	x0, #0                          ; =0x0
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ; -- Begin function task_balancer
lCPI1_0:
	.long	3                               ; 0x3
	.long	4                               ; 0x4
lCPI1_1:
	.long	5                               ; 0x5
	.long	6                               ; 0x6
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_task_balancer
	.p2align	2
_task_balancer:                         ; @task_balancer
	.cfi_startproc
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
	mov	w0, #32                         ; =0x20
	bl	_malloc
	ldp	w8, w9, [x19]
	mul	w8, w9, w8
	dup.2s	v0, w8
	lsl	w9, w8, #1
Lloh0:
	adrp	x10, lCPI1_0@PAGE
Lloh1:
	ldr	d1, [x10, lCPI1_0@PAGEOFF]
	lsl	w10, w8, #3
	sub	w10, w10, w8
	sbfx	w11, w8, #0, #29
	fmov	s2, w8
	mul.2s	v1, v0, v1
	mov.s	v2[1], w9
	mov.d	v2[1], v1[0]
	cmlt.4s	v1, v2, #0
	usra.4s	v2, v1, #29
	sshr.4s	v1, v2, #3
	str	q1, [x0]
Lloh2:
	adrp	x8, lCPI1_1@PAGE
Lloh3:
	ldr	d1, [x8, lCPI1_1@PAGEOFF]
	mul.2s	v0, v0, v1
	cmlt.2s	v1, v0, #0
	usra.2s	v0, v1, #29
	sshr.2s	v0, v0, #3
	str	d0, [x0, #16]
	add	w8, w10, #7
	cmp	w10, #0
	csel	w8, w8, w10, lt
	asr	w8, w8, #3
	stp	w8, w11, [x0, #24]
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
	.loh AdrpLdr	Lloh2, Lloh3
	.loh AdrpLdr	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ; -- Begin function main
lCPI2_0:
	.long	0                               ; 0x0
	.long	12500000                        ; 0xbebc20
lCPI2_1:
	.long	12500000                        ; 0xbebc20
	.long	25000000                        ; 0x17d7840
lCPI2_2:
	.long	25000000                        ; 0x17d7840
	.long	37500000                        ; 0x23c3460
lCPI2_3:
	.long	37500000                        ; 0x23c3460
	.long	50000000                        ; 0x2faf080
lCPI2_4:
	.long	50000000                        ; 0x2faf080
	.long	62500000                        ; 0x3b9aca0
lCPI2_5:
	.long	62500000                        ; 0x3b9aca0
	.long	75000000                        ; 0x47868c0
lCPI2_6:
	.long	75000000                        ; 0x47868c0
	.long	87500000                        ; 0x53724e0
lCPI2_7:
	.long	87500000                        ; 0x53724e0
	.long	100000000                       ; 0x5f5e100
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #352
	stp	d9, d8, [sp, #288]              ; 16-byte Folded Spill
	stp	x22, x21, [sp, #304]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #320]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #336]            ; 16-byte Folded Spill
	add	x29, sp, #336
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
Lloh4:
	adrp	x8, ___stack_chk_guard@GOTPAGE
Lloh5:
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
Lloh6:
	ldr	x8, [x8]
	stur	x8, [x29, #-56]
	mov	w0, #16                         ; =0x10
	bl	_malloc
	mov	x19, x0
	mov	w8, #10000                      ; =0x2710
	dup.2s	v0, w8
	str	d0, [x0]
	mov	w0, #1                          ; =0x1
	mov	w1, #33792                      ; =0x8400
	movk	w1, #6103, lsl #16
	bl	_calloc
	str	x0, [x19, #8]
Lloh7:
	adrp	x1, l_.memset_pattern@PAGE
Lloh8:
	add	x1, x1, l_.memset_pattern@PAGEOFF
	mov	w2, #33792                      ; =0x8400
	movk	w2, #6103, lsl #16
	bl	_memset_pattern16
	add	x1, sp, #8
	mov	w0, #6                          ; =0x6
	bl	_clock_gettime
	str	x19, [sp, #24]
Lloh9:
	adrp	x8, lCPI2_0@PAGE
Lloh10:
	ldr	d0, [x8, lCPI2_0@PAGEOFF]
	str	d0, [sp, #32]
	str	xzr, [sp, #40]
Lloh11:
	adrp	x2, _work@PAGE
Lloh12:
	add	x2, x2, _work@PAGEOFF
	sub	x21, x29, #120
	add	x20, sp, #24
	sub	x0, x29, #120
	add	x3, sp, #24
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_2
; %bb.1:
Lloh13:
	adrp	x8, ___stderrp@GOTPAGE
Lloh14:
	ldr	x8, [x8, ___stderrp@GOTPAGEOFF]
Lloh15:
	ldr	x3, [x8]
Lloh16:
	adrp	x0, l_.str.4@PAGE
Lloh17:
	add	x0, x0, l_.str.4@PAGEOFF
	mov	w1, #24                         ; =0x18
	mov	w2, #1                          ; =0x1
	bl	_fwrite
	b	LBB2_27
LBB2_2:
	str	x19, [sp, #48]
Lloh18:
	adrp	x8, lCPI2_1@PAGE
Lloh19:
	ldr	d0, [x8, lCPI2_1@PAGEOFF]
	str	d0, [sp, #56]
	str	xzr, [sp, #64]
Lloh20:
	adrp	x2, _work@PAGE
Lloh21:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #8
	add	x3, x20, #24
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_4
; %bb.3:
	mov	w20, #1                         ; =0x1
	b	LBB2_16
LBB2_4:
	add	x20, sp, #24
	str	x19, [sp, #72]
Lloh22:
	adrp	x8, lCPI2_2@PAGE
Lloh23:
	ldr	d0, [x8, lCPI2_2@PAGEOFF]
	str	d0, [sp, #80]
	str	xzr, [sp, #88]
Lloh24:
	adrp	x2, _work@PAGE
Lloh25:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #16
	add	x3, x20, #48
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_6
; %bb.5:
	mov	w20, #2                         ; =0x2
	b	LBB2_16
LBB2_6:
	str	x19, [sp, #96]
Lloh26:
	adrp	x8, lCPI2_3@PAGE
Lloh27:
	ldr	d0, [x8, lCPI2_3@PAGEOFF]
	str	d0, [sp, #104]
	str	xzr, [sp, #112]
Lloh28:
	adrp	x2, _work@PAGE
Lloh29:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #24
	add	x3, x20, #72
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_8
; %bb.7:
	mov	w20, #3                         ; =0x3
	b	LBB2_16
LBB2_8:
	add	x20, sp, #24
	str	x19, [sp, #120]
Lloh30:
	adrp	x8, lCPI2_4@PAGE
Lloh31:
	ldr	d0, [x8, lCPI2_4@PAGEOFF]
	str	d0, [sp, #128]
	str	xzr, [sp, #136]
Lloh32:
	adrp	x2, _work@PAGE
Lloh33:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #32
	add	x3, x20, #96
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_10
; %bb.9:
	mov	w20, #4                         ; =0x4
	b	LBB2_16
LBB2_10:
	str	x19, [sp, #144]
Lloh34:
	adrp	x8, lCPI2_5@PAGE
Lloh35:
	ldr	d0, [x8, lCPI2_5@PAGEOFF]
	str	d0, [sp, #152]
	str	xzr, [sp, #160]
Lloh36:
	adrp	x2, _work@PAGE
Lloh37:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #40
	add	x3, x20, #120
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_12
; %bb.11:
	mov	w20, #5                         ; =0x5
	b	LBB2_16
LBB2_12:
	add	x20, sp, #24
	str	x19, [sp, #168]
Lloh38:
	adrp	x8, lCPI2_6@PAGE
Lloh39:
	ldr	d0, [x8, lCPI2_6@PAGEOFF]
	str	d0, [sp, #176]
	str	xzr, [sp, #184]
Lloh40:
	adrp	x2, _work@PAGE
Lloh41:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #48
	add	x3, x20, #144
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_14
; %bb.13:
	mov	w20, #6                         ; =0x6
	b	LBB2_16
LBB2_14:
	str	x19, [sp, #192]
Lloh42:
	adrp	x8, lCPI2_7@PAGE
Lloh43:
	ldr	d0, [x8, lCPI2_7@PAGEOFF]
	str	d0, [sp, #200]
	str	xzr, [sp, #208]
Lloh44:
	adrp	x2, _work@PAGE
Lloh45:
	add	x2, x2, _work@PAGEOFF
	add	x0, x21, #56
	add	x3, x20, #168
	mov	x1, #0                          ; =0x0
	bl	_pthread_create
	cbz	w0, LBB2_29
; %bb.15:
	mov	w20, #7                         ; =0x7
LBB2_16:
Lloh46:
	adrp	x8, ___stderrp@GOTPAGE
Lloh47:
	ldr	x8, [x8, ___stderrp@GOTPAGEOFF]
Lloh48:
	ldr	x3, [x8]
Lloh49:
	adrp	x0, l_.str.4@PAGE
Lloh50:
	add	x0, x0, l_.str.4@PAGEOFF
	mov	w1, #24                         ; =0x18
	mov	w2, #1                          ; =0x1
	bl	_fwrite
LBB2_17:
	ldur	x0, [x29, #-120]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #40]
	movi.2d	v1, #0000000000000000
	fadd	d8, d0, d1
	cmp	w20, #1
	b.eq	LBB2_25
; %bb.18:
	ldur	x0, [x29, #-112]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #64]
	fadd	d8, d8, d0
	cmp	w20, #2
	b.eq	LBB2_25
; %bb.19:
	ldur	x0, [x29, #-104]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #88]
	fadd	d8, d8, d0
	cmp	w20, #3
	b.eq	LBB2_25
; %bb.20:
	ldur	x0, [x29, #-96]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #112]
	fadd	d8, d8, d0
	cmp	w20, #4
	b.eq	LBB2_25
; %bb.21:
	ldur	x0, [x29, #-88]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #136]
	fadd	d8, d8, d0
	cmp	w20, #5
	b.eq	LBB2_25
; %bb.22:
	ldur	x0, [x29, #-80]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #160]
	fadd	d8, d8, d0
	cmp	w20, #6
	b.eq	LBB2_25
; %bb.23:
	ldur	x0, [x29, #-72]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #184]
	fadd	d8, d8, d0
	cmp	w20, #7
	b.eq	LBB2_25
; %bb.24:
	ldur	x0, [x29, #-64]
	mov	x1, #0                          ; =0x0
	bl	_pthread_join
	ldr	d0, [sp, #208]
	fadd	d8, d8, d0
LBB2_25:
	cmp	w20, #8
	b.ne	LBB2_27
; %bb.26:
	str	d8, [sp]
Lloh51:
	adrp	x0, l_.str.5@PAGE
Lloh52:
	add	x0, x0, l_.str.5@PAGEOFF
	bl	_printf
LBB2_27:
	add	x1, sp, #24
	mov	w0, #6                          ; =0x6
	bl	_clock_gettime
	ldp	x8, x9, [sp, #24]
	ldp	x10, x11, [sp, #8]
	sub	x8, x8, x10
	scvtf	d0, x8
	sub	x8, x9, x11
	scvtf	d1, x8
	mov	x8, #225833675390976            ; =0xcd6500000000
	movk	x8, #16845, lsl #48
	fmov	d2, x8
	fdiv	d1, d1, d2
	fadd	d8, d1, d0
	mov	w8, #57600                      ; =0xe100
	movk	w8, #1525, lsl #16
	str	x8, [sp]
Lloh53:
	adrp	x0, l_.str.1@PAGE
Lloh54:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_printf
	str	d8, [sp]
Lloh55:
	adrp	x0, l_.str.2@PAGE
Lloh56:
	add	x0, x0, l_.str.2@PAGEOFF
	bl	_printf
	ldr	x0, [x19, #8]
	bl	_free
	mov	x0, x19
	bl	_free
	ldur	x8, [x29, #-56]
Lloh57:
	adrp	x9, ___stack_chk_guard@GOTPAGE
Lloh58:
	ldr	x9, [x9, ___stack_chk_guard@GOTPAGEOFF]
Lloh59:
	ldr	x9, [x9]
	cmp	x9, x8
	b.ne	LBB2_30
; %bb.28:
	mov	w0, #0                          ; =0x0
	ldp	x29, x30, [sp, #336]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #320]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #304]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #288]              ; 16-byte Folded Reload
	add	sp, sp, #352
	ret
LBB2_29:
	mov	w20, #8                         ; =0x8
	b	LBB2_17
LBB2_30:
	bl	___stack_chk_fail
	.loh AdrpAdd	Lloh11, Lloh12
	.loh AdrpLdr	Lloh9, Lloh10
	.loh AdrpAdd	Lloh7, Lloh8
	.loh AdrpLdrGotLdr	Lloh4, Lloh5, Lloh6
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpLdrGotLdr	Lloh13, Lloh14, Lloh15
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpLdr	Lloh18, Lloh19
	.loh AdrpAdd	Lloh24, Lloh25
	.loh AdrpLdr	Lloh22, Lloh23
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpLdr	Lloh26, Lloh27
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpLdr	Lloh30, Lloh31
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpLdr	Lloh34, Lloh35
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpLdr	Lloh38, Lloh39
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpLdr	Lloh42, Lloh43
	.loh AdrpAdd	Lloh49, Lloh50
	.loh AdrpLdrGotLdr	Lloh46, Lloh47, Lloh48
	.loh AdrpAdd	Lloh51, Lloh52
	.loh AdrpLdrGotLdr	Lloh57, Lloh58, Lloh59
	.loh AdrpAdd	Lloh55, Lloh56
	.loh AdrpAdd	Lloh53, Lloh54
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.1:                               ; @.str.1
	.asciz	"\354\233\220\354\206\214 \352\260\234\354\210\230: %zu\n"

l_.str.2:                               ; @.str.2
	.asciz	"\354\213\244\355\226\211 \354\213\234\352\260\204: %.6f\354\264\210\n"

l_.str.4:                               ; @.str.4
	.asciz	"\354\212\244\353\240\210\353\223\234 \354\203\235\354\204\261 \354\213\244\355\214\250\n"

l_.str.5:                               ; @.str.5
	.asciz	"\355\225\251\352\263\204: %.1f\n"

	.section	__TEXT,__literal16,16byte_literals
	.p2align	4, 0x0                          ; @.memset_pattern
l_.memset_pattern:
	.long	0x3f800000                      ; float 1
	.long	0x3f800000                      ; float 1
	.long	0x3f800000                      ; float 1
	.long	0x3f800000                      ; float 1

.subsections_via_symbols
