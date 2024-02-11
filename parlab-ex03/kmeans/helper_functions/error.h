#ifndef __ERROR_H
#define __ERROR_H

//#include "decls.h"

//__BEGIN_C_DECLS

void tab_print(int lvl);

void _printf(const char *fmt, va_list ap);

void warning(const char *fmt, ...);

void error(const char *fmt, ...);

void massert(bool condi, const char *fmt, ...);

void lprintf(short lvl, const char *fmt, ...);

//__END_C_DECLS

#endif  /* __ERROR_H */
