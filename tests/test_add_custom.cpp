// AddCustom 真机数值验证：z = x + y，覆盖 float32 与 float16
// 环境：Ascend 910B3 + CANN 9.0.0
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_add_custom.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                      \
  do {                                                                   \
    aclError _r = (expr);                                                \
    if (_r != ACL_SUCCESS) {                                             \
      printf("FAIL %-52s rc=%d (line %d)\n", #expr, (int)_r, __LINE__);  \
      return -1;                                                         \
    }                                                                    \
  } while (0)

// ---------------------------------------------------------------- float32
static int run_case_f32(size_t n) {
  const size_t bytes = n * sizeof(float);
  std::vector<float> hx(n), hy(n), hz(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    hx[i] = static_cast<float>(i % 97) * 0.5f;
    hy[i] = static_cast<float>(i % 89) * 0.25f;
  }

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

  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus st = aclnnAddCustomGetWorkspaceSize(tx, ty, tz, &wsSize, &executor);
  if (st != 0) { printf("  fp32 n=%-8zu FAIL GetWorkspaceSize rc=%d\n", n, (int)st); return -1; }
  void *workspace = nullptr;
  if (wsSize > 0) CHECK(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  st = aclnnAddCustom(workspace, wsSize, executor, stream);
  if (st != 0) { printf("  fp32 n=%-8zu FAIL aclnnAddCustom rc=%d\n", n, (int)st); return -1; }
  CHECK(aclrtSynchronizeStream(stream));
  CHECK(aclrtMemcpy(hz.data(), bytes, dz, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(hz[i] - (hx[i] + hy[i])) > 1e-5f) {
      if (bad < 3) printf("    mismatch idx=%zu got=%g expect=%g\n", i, hz[i], hx[i] + hy[i]);
      ++bad;
    }
  }
  printf("  fp32 n=%-8zu %s (mismatch=%zu)\n", n, bad == 0 ? "PASS" : "FAIL", bad);

  aclDestroyTensor(tx); aclDestroyTensor(ty); aclDestroyTensor(tz);
  aclrtFree(dx); aclrtFree(dy); aclrtFree(dz);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- float16
// 该 g++ 不支持 _Float16，用位运算做 fp32 <-> fp16（最近偶数舍入）转换
typedef uint16_t half_t;

static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t be = (x >> 23) & 0xffu;
  uint32_t m = x & 0x7fffffu;
  if (be == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
  int32_t he = (int32_t)be - 127 + 15;
  if (he >= 31) return (uint16_t)(sign | 0x7c00u);
  if (he <= 0) {
    if (he < -10) return (uint16_t)sign;
    m |= 0x800000u;
    const int32_t shift = 14 - he;
    uint32_t hm = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1u))) ++hm;
    return (uint16_t)(sign | hm);
  }
  uint32_t hm = m >> 13;
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
    ++hm;
    if (hm == 0x400u) { hm = 0; ++he; if (he >= 31) return (uint16_t)(sign | 0x7c00u); }
  }
  return (uint16_t)(sign | ((uint32_t)he << 10) | hm);
}

static float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1fu;
  const uint32_t m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) {
      x = sign;
    } else {  // 次正规数
      float v = std::ldexp((float)m, -24);
      return sign ? -v : v;
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | ((e - 15 + 127) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

static int run_case_f16(size_t n) {
  const size_t bytes = n * sizeof(half_t);
  std::vector<half_t> hx(n), hy(n), hz(n);
  std::vector<float> fx(n), fy(n);
  for (size_t i = 0; i < n; ++i) {
    fx[i] = static_cast<float>(i % 97) * 0.5f;
    fy[i] = static_cast<float>(i % 89) * 0.25f;
    hx[i] = f2h(fx[i]);
    hy[i] = f2h(fy[i]);
    hz[i] = 0;
  }

  aclrtStream stream = nullptr;
  CHECK(aclrtCreateStream(&stream));
  void *dx = nullptr, *dy = nullptr, *dz = nullptr;
  CHECK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dz, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(dx, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dy, bytes, hy.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

  int64_t dims[1] = {static_cast<int64_t>(n)};
  aclTensor *tx = aclCreateTensor(dims, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, dims, 1, dx);
  aclTensor *ty = aclCreateTensor(dims, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, dims, 1, dy);
  aclTensor *tz = aclCreateTensor(dims, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, dims, 1, dz);

  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus st = aclnnAddCustomGetWorkspaceSize(tx, ty, tz, &wsSize, &executor);
  if (st != 0) { printf("  fp16 n=%-8zu FAIL GetWorkspaceSize rc=%d\n", n, (int)st); return -1; }
  void *workspace = nullptr;
  if (wsSize > 0) CHECK(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  st = aclnnAddCustom(workspace, wsSize, executor, stream);
  if (st != 0) { printf("  fp16 n=%-8zu FAIL aclnnAddCustom rc=%d\n", n, (int)st); return -1; }
  CHECK(aclrtSynchronizeStream(stream));
  CHECK(aclrtMemcpy(hz.data(), bytes, dz, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

  // 期望值按「fp16 输入 + fp16 加法」估算；fp16 在 64 附近 ULP=0.0625，容差取 0.25
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float exp = h2f(hx[i]) + h2f(hy[i]);
    float got = h2f(hz[i]);
    if (std::fabs(got - exp) > 0.25f) {
      if (bad < 3) printf("    mismatch idx=%zu got=%f expect=%f\n", i, got, exp);
      ++bad;
    }
  }
  printf("  fp16 n=%-8zu %s (mismatch=%zu)\n", n, bad == 0 ? "PASS" : "FAIL", bad);

  aclDestroyTensor(tx); aclDestroyTensor(ty); aclDestroyTensor(tz);
  aclrtFree(dx); aclrtFree(dy); aclrtFree(dz);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  return bad == 0 ? 0 : 1;
}

int main() {
  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(0));

  const size_t cases[] = {6, 8, 1000, 1024, 1u << 20, 1000003};
  const size_t ncase = sizeof(cases) / sizeof(cases[0]);
  int failed = 0;

  printf("AddCustom on Ascend 910B3\n");
  for (size_t i = 0; i < ncase; ++i) {
    if (run_case_f32(cases[i]) != 0) ++failed;
  }
  for (size_t i = 0; i < ncase; ++i) {
    if (run_case_f16(cases[i]) != 0) ++failed;
  }

  aclrtResetDevice(0);
  aclFinalize();
  printf("SUMMARY: %s (%d/%zu cases failed)\n",
         failed == 0 ? "ALL PASS" : "HAS FAILURE", failed, ncase * 2);
  return failed == 0 ? 0 : 1;
}
