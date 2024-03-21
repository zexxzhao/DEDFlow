#ifndef __COMMON_H__
#define __COMMON_H__

#if defined(__cplusplus)
#define __BEGIN_DECLS__ extern "C" {
#define __END_DECLS__ }
#else
#define __BEGIN_DECLS__ /* empty */
#define __END_DECLS__ /* empty */
#endif

#ifdef __GNUC__ // GCC, Clang
#define assert(expr) while (!(expr)) __builtin_unreachable()
#else
#include <assert.h>
#endif


#include <stdint.h>
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

typedef char byte;
typedef ptrdiff_t size;


#endif // __COMMON_H__
