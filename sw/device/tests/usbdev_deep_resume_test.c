// Copyright lowRISC contributors (OpenTitan project).
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "sw/device/lib/testing/test_framework/ottf_main.h"
#include "sw/device/tests/usbdev_suspend.h"

OTTF_DEFINE_TEST_CONFIG();

bool test_main(void) {
  return usbdev_suspend_test(kSuspendPhaseDeepResume, kSuspendPhaseDeepResume,
                             1u, true);
}
