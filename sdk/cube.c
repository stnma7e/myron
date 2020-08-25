/*
# _____     ___ ____     ___ ____
#  ____|   |    ____|   |        | |____|
# |     ___|   |____ ___|    ____| |    \    PS2DEV Open Source Project.
#-----------------------------------------------------------------------
# (c) 2005 Naomi Peori <naomi@peori.ca>
# Licenced under Academic Free License version 2.0
# Review ps2sdk README & LICENSE files for further details.
#
*/

#include <kernel.h>
#include <stdlib.h>
#include <tamtypes.h>
#include <math3d.h>

#include <packet.h>

#include <dma_tags.h>
#include <gif_tags.h>
#include <gs_psm.h>

#include <dma.h>

#include <graph.h>

#include <draw.h>
#include <draw3d.h>

#include "mesh_data.c"

VECTOR object_position = { 0.00f, 0.00f, 0.00f, 1.00f };
VECTOR object_rotation = { 0.00f, 0.00f, 0.00f, 1.00f };

VECTOR camera_position = { 0.00f, 0.00f, 100.00f, 1.00f };
VECTOR camera_rotation = { 0.00f, 0.00f,   0.00f, 1.00f };

void init_gs(framebuffer_t *frame, zbuffer_t *z)
{

	// Define a 32-bit 640x512 framebuffer.
	frame->width = 640;
	frame->height = 512;
	frame->mask = 0;
	frame->psm = GS_PSM_32;
	frame->address = graph_vram_allocate(frame->width,frame->height, frame->psm, GRAPH_ALIGN_PAGE);

	// Enable the zbuffer.
	z->enable = DRAW_ENABLE;
	z->mask = 0;
	z->method = ZTEST_METHOD_GREATER_EQUAL;
	z->zsm = GS_ZBUF_32;
	z->address = graph_vram_allocate(frame->width,frame->height,z->zsm, GRAPH_ALIGN_PAGE);

	// Initialize the screen and tie the first framebuffer to the read circuits.
	graph_initialize(frame->address,frame->width,frame->height,frame->psm,0,0);

}

void init_drawing_environment(framebuffer_t *frame, zbuffer_t *z)
{

	packet_t *packet = packet_init(16,PACKET_NORMAL);

	// This is our generic qword pointer.
	qword_t *q = packet->data;

	// This will setup a default drawing environment.
	q = draw_setup_environment(q,0,frame,z);

	// Now reset the primitive origin to 2048-width/2,2048-height/2.
	q = draw_primitive_xyoffset(q,0,(2048-320),(2048-256));

	// Finish setting up the environment.
	q = draw_finish(q);

	// Now send the packet, no need to wait since it's the first.
	dma_channel_send_normal(DMA_CHANNEL_GIF,packet->data,q - packet->data, 0, 0);
	dma_wait_fast();

	packet_free(packet);

}

typedef struct {
	int context;

	// Matrices to setup the 3D environment and camera
	MATRIX local_world;
	MATRIX world_view;
	MATRIX view_screen;
	MATRIX local_screen;

	VECTOR *temp_vertices;

	prim_t prim;
	color_t color;

	xyz_t   *verts;
	color_t *colors;

	// The data packets for double buffering dma sends.
	packet_t *packets[2];
	qword_t *dmatag;
} render_ctx_t;

int render(render_ctx_t *rc)
{

	rc->packets[0] = packet_init(100,PACKET_NORMAL);
	rc->packets[1] = packet_init(100,PACKET_NORMAL);

	// Allocate calculation space.
	rc->temp_vertices = memalign(128, sizeof(VECTOR) * vertex_count);

	// Allocate register space.
	rc->verts  = memalign(128, sizeof(vertex_t) * vertex_count);
	rc->colors = memalign(128, sizeof(color_t)  * vertex_count);

	// Define the triangle primitive we want to use.
	rc->prim.type = PRIM_TRIANGLE;
	rc->prim.shading = PRIM_SHADE_GOURAUD;
	rc->prim.mapping = DRAW_DISABLE;
	rc->prim.fogging = DRAW_DISABLE;
	rc->prim.blending = DRAW_DISABLE;
	rc->prim.antialiasing = DRAW_ENABLE;
	rc->prim.mapping_type = PRIM_MAP_ST;
	rc->prim.colorfix = PRIM_UNFIXED;

	rc->color.r = 0x80;
	rc->color.g = 0x80;
	rc->color.b = 0x80;
	rc->color.a = 0x80;
	rc->color.q = 1.0f;

	// Create the view_screen matrix.
	create_view_screen(rc->view_screen, graph_aspect_ratio(), -3.00f, 3.00f, -3.00f, 3.00f, 1.00f, 2000.00f);

	// Wait for any previous dma transfers to finish before starting.
	dma_wait_fast();
}

int render_loop(framebuffer_t *frame, zbuffer_t *z, render_ctx_t *rc)
{
	int i;
	packet_t *current;
	qword_t *q;

    current = rc->packets[rc->context];

    // Spin the cube a bit.
    object_rotation[0] += 0.008f; //while (object_rotation[0] > 3.14f) { object_rotation[0] -= 6.28f; }
    object_rotation[1] += 0.012f; //while (object_rotation[1] > 3.14f) { object_rotation[1] -= 6.28f; }

    // Create the local_world matrix.
    create_local_world(rc->local_world, object_position, object_rotation);

    // Create the world_view matrix.
    create_world_view(rc->world_view, camera_position, camera_rotation);

    // Create the local_screen matrix.
    create_local_screen(rc->local_screen, rc->local_world, rc->world_view, rc->view_screen);

    // Calculate the vertex values.
    calculate_vertices(rc->temp_vertices, vertex_count, vertices, rc->local_screen);

    // Convert floating point vertices to fixed point and translate to center of screen.
    draw_convert_xyz(rc->verts, 2048, 2048, 32, vertex_count, (vertex_f_t*)rc->temp_vertices);

    // Convert floating point colours to fixed point.
    draw_convert_rgbq(rc->colors, vertex_count, (vertex_f_t*)rc->temp_vertices, (color_f_t*)colours, 0x80);

    // Grab our dmatag pointer for the dma chain.
    rc->dmatag = current->data;

    // Now grab our qword pointer and increment past the dmatag.
    q = rc->dmatag;
    q++;

    // Clear framebuffer but don't update zbuffer.
    q = draw_disable_tests(q,0,z);
    q = draw_clear(q,0,2048.0f-320.0f,2048.0f-256.0f,frame->width,frame->height,0x00,0x00,0x00);
    q = draw_enable_tests(q,0,z);

    // Draw the triangles using triangle primitive type.
    q = draw_prim_start(q,0,&rc->prim, &rc->color);

    for(i = 0; i < points_count; i++)
    {
        q->dw[0] = rc->colors[points[i]].rgbaq;
        q->dw[1] = rc->verts[points[i]].xyz;
        q++;
    }

    q = draw_prim_end(q,2,DRAW_RGBAQ_REGLIST);

    // Setup a finish event.
    q = draw_finish(q);

    // Define our dmatag for the dma chain.
    DMATAG_END(rc->dmatag,(q-current->data)-1,0,0,0);

    // Now send our current dma chain.
    dma_wait_fast();
    dma_channel_send_chain(DMA_CHANNEL_GIF,current->data, q - current->data, 0, 0);

    // Now switch our packets so we can process data while the DMAC is working.
    rc->context ^= 1;

    // Wait for scene to finish drawing
    draw_wait_finish();

    graph_wait_vsync();

	return 0;

}

int main(int argc, char **argv)
{

	// The buffers to be used.
	framebuffer_t frame;
	zbuffer_t z;

	// Init GIF dma channel.
	dma_channel_initialize(DMA_CHANNEL_GIF,NULL,0);
	dma_channel_fast_waits(DMA_CHANNEL_GIF);

	// Init the GS, framebuffer, and zbuffer.
	init_gs(&frame, &z);

	// Init the drawing environment and framebuffer.
	init_drawing_environment(&frame,&z);

	// Render the cube
    render_ctx_t rc;
    rc.context = 0;

	render(&rc);

    for (;;)
    {
        render_loop(&frame, &z, &rc);
    }

	packet_free(rc.packets[0]);
	packet_free(rc.packets[1]);


	// Sleep
	SleepThread();

	// End program.
	return 0;

}
