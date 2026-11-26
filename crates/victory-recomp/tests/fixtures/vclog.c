/* A log the images of the real-vcruntime140 C++ test share (the ordering of constructors, destructors and catch blocks). */
#define X __declspec(dllexport)
static int log_buf[64];
static int log_n;
X void vc_log(int v) {
  if (log_n < 64) log_buf[log_n++] = v;
}
X int vc_log_get(int *out) {
  for (int i = 0; i < log_n; i++) out[i] = log_buf[i];
  return log_n;
}
X void vc_log_reset(void) { log_n = 0; }
int DllMain(void *h, unsigned r, void *x) { return 1; }
