// The DLL half of the C++ exception test (own code, built for x86_64-pc-windows-msvc with exceptions on): throws and
// catches, with destructors in between, across the image boundary to cxx_a.cpp.
#define X extern "C" __declspec(dllexport)
#define I extern "C" __declspec(dllimport)
I void vc_log(int);

struct Guard {
  int id;
  Guard(int i) : id(i) { vc_log(i); }
  ~Guard() { vc_log(100 + id); }
};
struct Thing {
  int a, b;
};

X void b_throw_int(int v) { throw v; }
// The destructor of `g` runs while the exception unwinds this frame.
X void b_throw_with_guard(int v) {
  Guard g(1);
  throw v;
}
X void b_throw_thing(int v) {
  Thing t{v, v + 1};
  throw t;
}
// Catches the int, logs, and throws it again for whoever is above (in the other image).
X int b_rethrow(int v) {
  try {
    Guard g(2);
    throw v;
  } catch (int) {
    vc_log(50);
    throw;
  }
  return 0;
}
// An exception raised by `cb`, in the other image, caught here.
X int b_call_cb(int (*cb)(int), int v) {
  try {
    return cb(v);
  } catch (int e) {
    return e * 2;
  }
}
// No handler here: the destructor runs and the exception goes on.
X int b_nested(int v) {
  Guard g(3);
  b_throw_int(v);
  return 0;
}
extern "C" int DllMain(void *, unsigned, void *) { return 1; }
