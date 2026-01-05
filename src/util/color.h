#pragma once

#include <stdio.h>
#include <stdarg.h>

void blue(const char *fmt, ...) {
    flockfile(stdout);

    va_list args;
    va_start(args, fmt);

    printf("\033[0;34m");      // set color to blue
    vprintf(fmt, args);        // print with format string
    printf("\033[0m\n");       // reset color and add newline

    va_end(args);
    funlockfile(stdout);
}

void yellow(const char *fmt, ...) {
    flockfile(stdout);

    va_list args;
    va_start(args, fmt);

    printf("\033[0;33m");      // yellow
    vprintf(fmt, args);
    printf("\033[0m\n");

    va_end(args);
    funlockfile(stdout);
}

void magenta(const char *fmt, ...) {
    flockfile(stdout);

    va_list args;
    va_start(args, fmt);

    printf("\033[0;35m");      // magenta
    vprintf(fmt, args);
    printf("\033[0m\n");

    va_end(args);
    funlockfile(stdout);
}
void green(const char *fmt, ...) {
    flockfile(stdout);

    va_list args;
    va_start(args, fmt);

    printf("\033[0;32m");      // green
    vprintf(fmt, args);
    printf("\033[0m\n");

    va_end(args);
    funlockfile(stdout);

}

void red(const char *fmt, ...) {
    flockfile(stdout);

    va_list args;
    va_start(args, fmt);

    printf("\033[0;31m");      // red
    vprintf(fmt, args);
    printf("\033[0m\n");

    va_end(args);
    funlockfile(stdout);

}
