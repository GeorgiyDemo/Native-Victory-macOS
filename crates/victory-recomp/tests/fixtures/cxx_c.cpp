// C++ exception cases for the real vcruntime140 (own code, MSVC target): a try inside a catch, a rethrow, destructors
// of several frames during one unwind, catch by base class, a thrown object with a destructor and a copy constructor.
#define X extern "C" __declspec(dllexport)
#define I extern "C" __declspec(dllimport)
I void vc_log(int);
struct Guard {
  int id;
  Guard(int i) : id(i) { vc_log(i); }
  ~Guard() { vc_log(100 + id); }
};
struct Base {
  int x;
};
struct Derived : Base {
  int y;
};
struct Obj {
  int v;
  Obj(int a) : v(a) { vc_log(300 + a); }
  Obj(const Obj &o) : v(o.v) { vc_log(400 + v); }
  ~Obj() { vc_log(500 + v); }
};

__declspec(noinline) static void thrower(int v) {
  Guard g(10);
  throw v;
}
__declspec(noinline) static void deeper(int v) {
  Guard g(20);
  thrower(v);
}
// destructors of three frames, nearest first, then the catch block
X int c_unwind(int v) {
  try {
    Guard g(30);
    deeper(v);
  } catch (int e) {
    vc_log(900);
    return e;
  }
  return -1;
}
// a try inside a catch block that throws and catches again, then the outer catch's code goes on
X int c_nested(int v) {
  int r = 0;
  try {
    throw v;
  } catch (int e) {
    vc_log(910);
    try {
      throw 2.5;
    } catch (double d) {
      vc_log(911);
      r = e + (int)d;
    }
    vc_log(912);
  }
  return r;
}
// a throw from inside a catch block, caught outside it
X int c_throw_in_catch(int v) {
  try {
    try {
      throw v;
    } catch (int e) {
      Guard g(40);
      throw e + 1;
    }
  } catch (int e) {
    vc_log(920);
    return e;
  }
  return -1;
}
// rethrow
X int c_rethrow(int v) {
  try {
    try {
      throw v;
    } catch (int) {
      vc_log(930);
      throw;
    }
  } catch (int e) {
    vc_log(931);
    return e * 2;
  }
  return -1;
}
// catch by a base class
X int c_base(int v) {
  try {
    Derived d;
    d.x = v;
    d.y = 7;
    throw d;
  } catch (const Base &b) {
    return b.x + 1000;
  }
  return -1;
}
// a thrown object that is copied into the exception and destroyed after the catch
X int c_object(int v) {
  try {
    throw Obj(v);
  } catch (const Obj &o) {
    vc_log(940);
    return o.v;
  }
  return -1;
}
extern "C" int DllMain(void *, unsigned, void *) { return 1; }
