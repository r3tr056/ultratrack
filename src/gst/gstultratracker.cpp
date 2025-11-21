#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <string>
#include "ultratrack_manager.hpp"

// DeepStream SDK headers
#ifdef HAS_DEEPSTREAM
#include "gstnvdsmeta.h"
#include "nvds_meta.h"
#include "nvbufsurface.h"
#else
// Stub definitions if DeepStream not available
struct NvBufSurface {
    uint32_t batchSize;
    void* surfaceList;
};
#endif

#define GST_TYPE_ULTRATRACK (gst_ultratrack_get_type())
#define GST_ULTRATRACK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_ULTRATRACK, GstUltraTrack))

typedef struct _GstUltraTrack GstUltraTrack;
typedef struct _GstUltraTrackClass GstUltraTrackClass;

struct _GstUltraTrack {
    GstBaseTransform element;
    UltraTrackManager* tracker;
    gchar* gallery_path;
    gchar* nanotrack_engine;
    gchar* yolo_engine;
    gchar* osnet_engine;
    guint frame_count;
};

struct _GstUltraTrackClass {
    GstBaseTransformClass parent_class;
};

enum {
    PROP_0,
    PROP_GALLERY,
    PROP_NANOTRACK_ENGINE,
    PROP_YOLO_ENGINE,
    PROP_OSNET_ENGINE
};

G_DEFINE_TYPE(GstUltraTrack, gst_ultratrack, GST_TYPE_BASE_TRANSFORM);

static void gst_ultratrack_finalize(GObject* object);
static void gst_ultratrack_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec);
static void gst_ultratrack_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec);
static GstFlowReturn gst_ultratrack_transform_ip(GstBaseTransform* trans, GstBuffer* buf);
static gboolean gst_ultratrack_start(GstBaseTransform* trans);
static gboolean gst_ultratrack_stop(GstBaseTransform* trans);

static void gst_ultratrack_class_init(GstUltraTrackClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);

    gobject_class->set_property = gst_ultratrack_set_property;
    gobject_class->get_property = gst_ultratrack_get_property;
    gobject_class->finalize = gst_ultratrack_finalize;
    
    base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_ultratrack_transform_ip);
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_ultratrack_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_ultratrack_stop);
    
    // Properties
    g_object_class_install_property(gobject_class, PROP_GALLERY,
        g_param_spec_string("target-gallery", "Target Gallery",
                            "Path to folder containing 2-5 images of the target object",
                            NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_NANOTRACK_ENGINE,
        g_param_spec_string("nanotrack-engine", "NanoTrack Engine",
                            "Path to NanoTrack TensorRT engine file",
                            "models/nanotrack.engine", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_YOLO_ENGINE,
        g_param_spec_string("yolo-engine", "YOLO Engine",
                            "Path to YOLO TensorRT engine file",
                            "models/yolov11.engine", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_OSNET_ENGINE,
        g_param_spec_string("osnet-engine", "OSNet Engine",
                            "Path to OSNet TensorRT engine file (DLA)",
                            "models/osnet.engine", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    // Element metadata
    gst_element_class_set_static_metadata(element_class,
        "UltraTrack Zero-Copy Tracker",
        "Filter/Effect/Video",
        "High-performance object tracking with few-shot learning",
        "UltraTrack Team");
    
    // Pad templates (NVMM memory)
    GstCaps* caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    GstPadTemplate* src_template = gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps);
    GstPadTemplate* sink_template = gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps);
    gst_element_class_add_pad_template(element_class, src_template);
    gst_element_class_add_pad_template(element_class, sink_template);
    gst_caps_unref(caps);
}

static void gst_ultratrack_init(GstUltraTrack* self) {
    self->tracker = nullptr;
    self->gallery_path = nullptr;
    self->nanotrack_engine = g_strdup("models/nanotrack.engine");
    self->yolo_engine = g_strdup("models/yolov11.engine");
    self->osnet_engine = g_strdup("models/osnet.engine");
    self->frame_count = 0;
}

static void gst_ultratrack_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstUltraTrack* self = GST_ULTRATRACK(object);
    
    switch (prop_id) {
        case PROP_GALLERY:
            g_free(self->gallery_path);
            self->gallery_path = g_value_dup_string(value);
            if (self->tracker && self->gallery_path) {
                self->tracker->load_gallery_from_disk(self->gallery_path);
            }
            break;
        case PROP_NANOTRACK_ENGINE:
            g_free(self->nanotrack_engine);
            self->nanotrack_engine = g_value_dup_string(value);
            break;
        case PROP_YOLO_ENGINE:
            g_free(self->yolo_engine);
            self->yolo_engine = g_value_dup_string(value);
            break;
        case PROP_OSNET_ENGINE:
            g_free(self->osnet_engine);
            self->osnet_engine = g_value_dup_string(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ultratrack_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstUltraTrack* self = GST_ULTRATRACK(object);
    
    switch (prop_id) {
        case PROP_GALLERY:
            g_value_set_string(value, self->gallery_path);
            break;
        case PROP_NANOTRACK_ENGINE:
            g_value_set_string(value, self->nanotrack_engine);
            break;
        case PROP_YOLO_ENGINE:
            g_value_set_string(value, self->yolo_engine);
            break;
        case PROP_OSNET_ENGINE:
            g_value_set_string(value, self->osnet_engine);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ultratrack_finalize(GObject* object) {
    GstUltraTrack* self = GST_ULTRATRACK(object);
    g_free(self->gallery_path);
    g_free(self->nanotrack_engine);
    g_free(self->yolo_engine);
    g_free(self->osnet_engine);
    G_OBJECT_CLASS(gst_ultratrack_parent_class)->finalize(object);
}

static gboolean gst_ultratrack_start(GstBaseTransform* trans) {
    GstUltraTrack* self = GST_ULTRATRACK(trans);
    
    UltraTrackManager::Config config;
    config.nanotrack_engine = self->nanotrack_engine ? self->nanotrack_engine : "models/nanotrack.engine";
    config.yolo_engine = self->yolo_engine ? self->yolo_engine : "models/yolov11.engine";
    config.osnet_engine = self->osnet_engine ? self->osnet_engine : "models/osnet.engine";
    
    try {
        self->tracker = new UltraTrackManager(config);
        
        if (self->gallery_path) {
            self->tracker->load_gallery_from_disk(self->gallery_path);
        }
        
        GST_INFO_OBJECT(self, "UltraTrackManager initialized successfully");
        return TRUE;
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT(self, "Failed to initialize UltraTrackManager: %s", e.what());
        return FALSE;
    }
}

static gboolean gst_ultratrack_stop(GstBaseTransform* trans) {
    GstUltraTrack* self = GST_ULTRATRACK(trans);
    if (self->tracker) {
        delete self->tracker;
        self->tracker = nullptr;
    }
    return TRUE;
}

static GstFlowReturn gst_ultratrack_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstUltraTrack* self = GST_ULTRATRACK(trans);
    GstMapInfo in_map_info;
    NvBufSurface* surface = NULL;

    // 1. Map the buffer to get NvBufSurface
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
        return GST_FLOW_ERROR;
    }
    
    surface = (NvBufSurface*)in_map_info.data;

    if (self->tracker && surface) {
        // 2. Process Batch
        self->tracker->process_batch(surface);
        
#ifdef HAS_DEEPSTREAM
        // 3. Attach metadata
        NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
        if (!batch_meta) {
            batch_meta = nvds_create_batch_meta(surface->batchSize);
            gst_buffer_add_nvds_batch_meta(buf, batch_meta);
        }
        
        // Get the tracker's current bbox
        // In real implementation, UltraTrackManager would return this via get_tracks()
        // For now, we demonstrate the structure
        
        NvDsFrameMeta* frame_meta = nvds_create_frame_meta(batch_meta);
        nvds_add_frame_meta_to_batch(batch_meta, frame_meta);
        
        NvDsObjectMeta* obj_meta = nvds_create_obj_meta(batch_meta);
        obj_meta->class_id = 0;
        obj_meta->object_id = 1; // Persistent ID
        obj_meta->confidence = 0.9f;
        
        // Set bbox (would come from tracker)
        obj_meta->rect_params.left = 0;
        obj_meta->rect_params.top = 0;
        obj_meta->rect_params.width = 100;
        obj_meta->rect_params.height = 100;
        
        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, nullptr);
#endif
    }

    self->frame_count++;
    gst_buffer_unmap(buf, &in_map_info);
    return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "ultratracker", GST_RANK_NONE, GST_TYPE_ULTRATRACK);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    ultratracker,
    "UltraTrack Zero-Copy Plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "UltraTrack",
    "https://ultratrack.io/"
)
