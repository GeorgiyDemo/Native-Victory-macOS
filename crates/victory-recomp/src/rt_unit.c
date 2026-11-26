/* What a lifted image built as a dylib of its own (beside the runtime, which is built with -DRT_MULTI) defines:
 * its own IMG and the unit record the runtime registers it by (rt_image_add). Not used by a single-image build,
 * where rt_host.c is compiled in with the lifted code. */
#include "rt.h"
u8 *IMG;
const RtUnit rt_unit = { &IMG, rt_table, &rt_table_n };
