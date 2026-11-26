// The "exe" half of the C++ exception test (own code, built for x86_64-pc-windows-msvc with exceptions on): catches what
// cxx_b.cpp throws, by type, by `...` and after a rethrow, and throws for cxx_b.cpp to catch through a pointer.
#define X extern "C" __declspec(dllexport)
#define I extern "C" __declspec(dllimport)
I void vc_log(int);
I void b_throw_int(int);
I void b_throw_with_guard(int);
I void b_throw_thing(int);
I int b_rethrow(int);
I int b_call_cb(int (*)(int), int);
I int b_nested(int);

struct Guard {
  int id;
  Guard(int i) : id(i) { vc_log(i); }
  ~Guard() { vc_log(100 + id); }
};
struct Thing {
  int a, b;
};

X int a_catch_int(int v) {
  try {
    b_throw_int(v);
  } catch (int e) {
    return e + 1;
  }
  return -1;
}
// Destructors of both images run, the one nearer the throw first, and before the catch block.
X int a_dtors(int v) {
  try {
    Guard g(5);
    b_throw_with_guard(v);
  } catch (int e) {
    vc_log(777);
    return e;
  }
  return -1;
}
X int a_catch_all(int v) {
  try {
    b_throw_with_guard(v);
  } catch (...) {
    vc_log(888);
    return 42;
  }
  return -1;
}
X int a_rethrown(int v) {
  try {
    return b_rethrow(v);
  } catch (int e) {
    return e + 1000;
  }
}
X int a_thing(int v) {
  try {
    b_throw_thing(v);
  } catch (const Thing &t) {
    return t.a * 10 + t.b;
  }
  return -1;
}
// The inner catch (int) does not take a double; the outer catch (...) does.
X int a_mismatch(void) {
  try {
    try {
      throw 2.5;
    } catch (int) {
      return 1;
    }
  } catch (...) {
    return 5;
  }
}
static int cb_throw(int v) { throw v + 1; }
// Thrown here, caught in the DLL, through a function pointer.
X int a_through_b(int v) { return b_call_cb(cb_throw, v); }
X int a_nested(int v) {
  try {
    return b_nested(v);
  } catch (int e) {
    return e + 2000;
  }
}
extern "C" int DllMain(void *, unsigned, void *) { return 1; }
