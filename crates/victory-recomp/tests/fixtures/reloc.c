/* Linked into the arm64 host test images so each has a base relocation (a pointer to its own data): a
 * freestanding image without one cannot be moved, and 0x180000000, the default base, is the dyld shared cache
 * on arm64 macOS. */
static int anchor;
int *volatile reloc_anchor = &anchor;
