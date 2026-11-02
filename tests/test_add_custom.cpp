// AddCustom 真机数值验证：z = x + y (float32)，覆盖小 shape / 单 tile / 多核多 tile / 非对齐尾部
// 运行环境：Ascend 910B3 + CANN 9.0.0
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_add_custom.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr)                                                      \
  do {                                                                   \
    aclError _r = (expr);                                                \
    if (_r != ACL_SUCCESS) {                                             \
      printf("FAIL %-52s rc=%d (line %d)\n", #expr, (int)_r, __LINE__);  \
      return -1;                                                         \
    }                                                                    \
  } while (0)

static int run_case(size_t n) {
  const size_t bytes = n * sizeof(float);
  std::vector<float> hx(n), hy(n), hz(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    hx[i] = static_cast<float>(i % 97) * 0.5f;
    hy[i] = static_cast<float>(i % 89) * 0.25f;
  }

  CHECK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK(aclrtCreateStream(&stream));

  void *dx = nullptr, *dy = nullptr, *dz = nullptr;
  CHECK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dz, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(dx, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dy, bytes, hy.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

  int64_t dims[1] = {static_cast<int64_t>(n)};
  aclTensor *tx = aclCreateTensor(dims, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dims, 1, dx);
  aclTensor *ty = aclCreateTensor(dims, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dims, 1, dy);
  aclTensor *tz = aclCreateTensor(dims, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dims, 1, dz);
  if (!tx || !ty || !tz) { printf("FAIL tensor create\n"); return -1; }

  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus st = aclnnAddCustomGetWorkspaceSize(tx, ty, tz, &wsSize, &executor);
  if (st != 0) { printf("  n=%-8zu FAIL GetWorkspaceSize rc=%d\n", n, (int)st); return -1; }

  void *workspace = nullptr;
  if (wsSize > 0) CHECK(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  st = aclnnAddCustom(workspace, wsSize, executor, stream);
  if (st != 0) { printf("  n=%-8zu FAIL aclnnAddCustom rc=%d\n", n, (int)st); return -1; }

  CHECK(aclrtSynchronizeStream(stream));
  CHECK(aclrtMemcpy(hz.data(), bytes, dz, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(hz[i] - (hx[i] + hy[i])) > 1e-5f) {
      if (bad < 3) printf("    mismatch idx=%zu got=%f expect=%f\n", i, hz[i], hx[i] + hy[i]);
      ++bad;
    }
  }
  printf("  n=%-8zu %s (mismatch=%zu)\n", n, bad == 0 ? "PASS" : "FAIL", bad);

  aclDestroyTensor(tx); aclDestroyTensor(ty); aclDestroyTensor(tz);
  aclrtFree(dx); aclrtFree(dy); aclrtFree(dz);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  return bad == 0 ? 0 : 1;
}

int main() {
  CHECK(aclInit(nullptr));

  const size_t cases[] = {6, 8, 1000, 1024, 1u << 20, 1000003};
  int failed = 0;
  printf("AddCustom on Ascend 910B3, float32\n");
  for (size_t n : cases) {
    if (run_case(n) != 0) ++failed;
  }

  aclrtResetDevice(0);
  aclFinalize();
  printf("SUMMARY: %s (%d/%zu cases failed)\n",
         failed == 0 ? "ALL PASS" : "HAS FAILURE", failed, sizeof(cases) / sizeof(cases[0]));
  return failed == 0 ? 0 : 1;
}
