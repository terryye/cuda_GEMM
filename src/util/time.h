#pragma once

#include <time.h>

// Wall clock timer functions
void start_timer(struct timespec *start) {
    clock_gettime(CLOCK_MONOTONIC, start);
}

double stop_timer(struct timespec *start) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Method 1: Handle nanosecond borrowing manually
    long sec_diff = end.tv_sec - start->tv_sec;
    long nsec_diff = end.tv_nsec - start->tv_nsec;

    if (nsec_diff < 0) {
        sec_diff--;
        nsec_diff += 1000000000L;  // Add 1 second worth of nanoseconds
    }

    return (double)sec_diff + (double)nsec_diff / 1e9;
}
