allocator: std.mem.Allocator,
file: std.fs.File,
file_buf: []align(std.mem.page_size) u8,
ggml_mem_buffer: []u8,

hparams: HParams,

// final normalization
ln_f_g: *c.ggml_tensor,
ln_f_b: *c.ggml_tensor,

wte: *c.ggml_tensor, // word embedding

lmh_g: *c.ggml_tensor, // language model head
//  lmh_b: *c.ggml_tensor ; // language model bias

layers: []Layer,

// key + value memory
memory_k: *c.ggml_tensor,
memory_v: *c.ggml_tensor,

//
ctx: *c.ggml_context,

/// Contains all of the tokens in one big slice
tokens_arena: std.heap.ArenaAllocator,
/// Token ID == location in array hashmap
tokens: std.StringArrayHashMapUnmanaged(void),

const Model = @This();

// default hparams (GPT-J 6B)
pub const HParams = struct {
    n_vocab: u32 = 50400,
    n_ctx: u32 = 2048,
    n_embd: u32 = 4096,
    n_head: u32 = 16,
    n_layer: u32 = 28,
    n_rot: u32 = 64,
    use_parallel_residual: u32 = 1, // 1 = true, 0 = false
    f16: u32 = 1,
};

pub const Layer = struct {
    // input_layernorm
    input_layernorm_weight: *c.ggml_tensor,
    input_layernorm_bias: *c.ggml_tensor,

    // post_attention_layernorm
    post_attention_layernorm_weight: *c.ggml_tensor,
    post_attention_layernorm_bias: *c.ggml_tensor,

    // attention
    c_attn_q_proj_w: *c.ggml_tensor,
    c_attn_k_proj_w: *c.ggml_tensor,
    c_attn_v_proj_w: *c.ggml_tensor,

    c_attn_q_proj_bias: *c.ggml_tensor,
    c_attn_k_proj_bias: *c.ggml_tensor,
    c_attn_v_proj_bias: *c.ggml_tensor,

    c_attn_proj_w: *c.ggml_tensor,
    c_attn_proj_bias: *c.ggml_tensor,

    // ff
    c_mlp_fc_w: *c.ggml_tensor,
    c_mlp_fc_b: *c.ggml_tensor,

    c_mlp_proj_w_trans: *c.ggml_tensor,
    c_mlp_proj_b: *c.ggml_tensor,
};

// load the model's weights from a file
pub fn load(allocator: std.mem.Allocator, filename: []const u8, n_ctx: u32) !Model {
    std.log.info("{s}: loading model from \"{}\"...", .{ @src().fn_name, std.zig.fmtEscapes(filename) });

    var progress = std.Progress{};
    var root_progress_node = progress.start(filename, 0);
    root_progress_node.activate();
    defer root_progress_node.end();

    var model: Model = undefined;
    model.allocator = allocator;

    model.file = std.fs.cwd().openFile(filename, .{}) catch |err| {
        std.log.warn("{s}: failed to open \"{}\": {}", .{ @src().fn_name, std.zig.fmtEscapes(filename), err });
        return err;
    };
    errdefer model.file.close();
    const file_len = try model.file.getEndPos();

    const file_buf = try std.os.mmap(null, file_len, std.os.PROT.READ, std.os.MAP.SHARED, model.file.handle, 0);
    errdefer std.os.munmap(file_buf);

    // verify magic
    {
        const magic = try model.file.reader().readIntLittle(u32);
        if (magic != 0x67676d6c) {
            std.log.warn("{s}: invalid model file \"{}\" (bad magic)", .{ @src().fn_name, std.zig.fmtEscapes(filename) });
            return error.BadMagic;
        }
    }

    // load hparams

    model.hparams.n_vocab = try model.file.reader().readIntLittle(u32);
    model.hparams.n_embd = try model.file.reader().readIntLittle(u32);
    model.hparams.n_head = try model.file.reader().readIntLittle(u32);
    model.hparams.n_layer = try model.file.reader().readIntLittle(u32);
    model.hparams.n_rot = try model.file.reader().readIntLittle(u32);
    model.hparams.use_parallel_residual = try model.file.reader().readIntLittle(u32);
    model.hparams.f16 = try model.file.reader().readIntLittle(u32);

    model.hparams.n_ctx = n_ctx;

    std.log.info("{s}: n_vocab = {}", .{ @src().fn_name, model.hparams.n_vocab });
    std.log.info("{s}: n_ctx   = {}", .{ @src().fn_name, model.hparams.n_ctx });
    std.log.info("{s}: n_embd  = {}", .{ @src().fn_name, model.hparams.n_embd });
    std.log.info("{s}: n_head  = {}", .{ @src().fn_name, model.hparams.n_head });
    std.log.info("{s}: n_layer = {}", .{ @src().fn_name, model.hparams.n_layer });
    std.log.info("{s}: n_rot   = {}", .{ @src().fn_name, model.hparams.n_rot });
    std.log.info("{s}: use_parallel_residual = {}", .{ @src().fn_name, model.hparams.use_parallel_residual });
    std.log.info("{s}: f16     = {}", .{ @src().fn_name, model.hparams.f16 });

    // load vocab
    model.tokens = .{};
    errdefer model.tokens.deinit(allocator);
    model.tokens_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer model.tokens_arena.deinit();
    for (0..model.hparams.n_vocab) |_| {
        const len = try model.file.reader().readIntLittle(u32);

        const word = try model.tokens_arena.allocator().alloc(u8, len);
        try model.file.reader().readNoEof(word);

        try model.tokens.put(allocator, word, {});
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    const wtype: c.enum_ggml_type = switch (model.hparams.f16) {
        0 => c.GGML_TYPE_F32,
        1 => c.GGML_TYPE_F16,
        2 => c.GGML_TYPE_Q4_0,
        3 => c.GGML_TYPE_Q4_1,
        else => |w| {
            std.log.warn("{s}: invalid model model.file \"{}\" (bad f16 value {d})", .{ @src().fn_name, std.zig.fmtEscapes(filename), w });
            return error.BadF16Value;
        },
    };

    var ctx_size: usize = 0;

    {
        if (false) ctx_size += model.hparams.n_embd * c.ggml_type_size(c.c.GGML_TYPE_F32); // ln_f_g
        if (false) ctx_size += model.hparams.n_embd * c.ggml_type_size(c.c.GGML_TYPE_F32); // ln_f_b

        if (false) ctx_size += model.hparams.n_embd * model.hparams.n_vocab * c.ggml_type_size(wtype); // wte

        if (false) ctx_size += model.hparams.n_embd * model.hparams.n_vocab * c.ggml_type_size(wtype); // lmh_g
        // ctx_size +=        n_vocab*c.ggml_type_sizef(c.GGML_TYPE_F32); // lmh_b

        if (false) { // Transformer layers
            { // Layernorms
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // input_layernorm_weight
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // input_layernorm_bias

                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // post_attention_layernorm_weight
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // post_attention_layernorm_bias
            }

            { // Attention layer
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_attn_q_proj_w
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_attn_k_proj_w
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_attn_v_proj_w

                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_attn_q_proj_bias
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_attn_k_proj_bias
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_attn_v_proj_bias

                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_attn_proj_w
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_attn_proj_bias
            }

            { // Feedforward layer
                ctx_size += model.hparams.n_layer * (4 * model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_mlp_fc_w
                ctx_size += model.hparams.n_layer * (4 * model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_mlp_fc_b

                ctx_size += model.hparams.n_layer * (4 * model.hparams.n_embd * model.hparams.n_embd * c.ggml_type_size(wtype)); // c_mlp_proj_w_trans
                ctx_size += model.hparams.n_layer * (model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32)); // c_mlp_proj_b
            }
        }

        if (false) ctx_size += model.hparams.n_ctx * model.hparams.n_layer * model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32); // memory_k
        if (false) ctx_size += model.hparams.n_ctx * model.hparams.n_layer * model.hparams.n_embd * c.ggml_type_size(c.GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 16 * model.hparams.n_layer) * 256; // object overhead

        std.log.info("{s}: ggml ctx size = {d:6.2} MB", .{ @src().fn_name, @intToFloat(f32, ctx_size) / (1024 * 1024) });
    }

    // create the ggml context
    model.ggml_mem_buffer = try allocator.alloc(u8, ctx_size);
    errdefer allocator.free(model.ggml_mem_buffer);
    var params = c.ggml_init_params{
        .mem_size = model.ggml_mem_buffer.len,
        .mem_buffer = model.ggml_mem_buffer.ptr,
        .no_alloc = true,
    };

    model.ctx = c.ggml_init(params) orelse {
        std.log.warn("{s}: ggml_init() failed", .{@src().fn_name});
        return error.GGMLInit;
    };

    // prepare memory for the weights
    model.layers = try allocator.alloc(Layer, model.hparams.n_layer);
    errdefer allocator.free(model.layers);
    {
        model.wte = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_vocab);

        model.ln_f_g = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
        model.ln_f_b = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);

        model.lmh_g = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_vocab);
        // model.lmh_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

        for (model.layers) |*layer| {
            // Layernorms
            layer.input_layernorm_weight = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
            layer.input_layernorm_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
            layer.post_attention_layernorm_weight = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
            layer.post_attention_layernorm_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);

            // Attention
            layer.c_attn_q_proj_w = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_embd);
            layer.c_attn_k_proj_w = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_embd);
            layer.c_attn_v_proj_w = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_embd);

            layer.c_attn_q_proj_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
            layer.c_attn_k_proj_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
            layer.c_attn_v_proj_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);

            layer.c_attn_proj_w = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, model.hparams.n_embd);
            layer.c_attn_proj_bias = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);

            // Feedforward
            layer.c_mlp_fc_w = c.ggml_new_tensor_2d(model.ctx, wtype, model.hparams.n_embd, 4 * model.hparams.n_embd);
            layer.c_mlp_fc_b = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, 4 * model.hparams.n_embd);

            layer.c_mlp_proj_w_trans = c.ggml_new_tensor_2d(model.ctx, wtype, 4 * model.hparams.n_embd, model.hparams.n_embd);
            layer.c_mlp_proj_b = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, model.hparams.n_embd);
        }
    }

    // key + value memory
    {
        const n_mem = model.hparams.n_layer * model.hparams.n_ctx;
        const n_elements = model.hparams.n_embd * n_mem;

        model.memory_k = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, n_elements);
        model.memory_v = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, n_elements);

        const memory_size = c.ggml_nbytes(model.memory_k) + c.ggml_nbytes(model.memory_v);

        std.log.info("{s}: memory size = {d:8.2} MB, n_mem = {d}", .{ @src().fn_name, @intToFloat(f32, memory_size) / (1024 * 1024), n_mem });
    }

    // load weights
    {
        var n_tensors: usize = 0;
        var total_size: usize = 0;

        var name_buffer = std.ArrayListUnmanaged(u8){};
        defer name_buffer.deinit(allocator);

        while (true) {
            if (try model.file.getPos() == try model.file.getEndPos()) {
                break;
            }

            const n_dims = try model.file.reader().readIntLittle(u32);
            const name_length = try model.file.reader().readIntLittle(u32);
            const ftype = try model.file.reader().readIntLittle(u32);

            if (n_dims > 2) {
                std.log.warn("{s}: n_dims greater than 2 unimplemented, n_dims = {d}", .{ @src().fn_name, n_dims });
                return error.Unimplemented;
            }

            var nelements: u32 = 1;
            var shape = [2]u32{ 1, 1 };
            for (shape[0..n_dims]) |*ne| {
                ne.* = try model.file.reader().readIntLittle(u32);
                nelements *= ne.*;
            }

            try name_buffer.resize(allocator, @intCast(usize, name_length));
            try model.file.reader().readNoEof(name_buffer.items);

            var tensor_progress_node = progress.start(name_buffer.items, 0);
            tensor_progress_node.activate();
            defer tensor_progress_node.end();

            // TODO: Parse name
            var tensor: ?**c.ggml_tensor = null;

            var name_segment_iter = std.mem.split(u8, name_buffer.items, ".");
            _ = name_segment_iter.next() orelse return error.BadLayerName; // gpt_neox
            const layer_or_other = name_segment_iter.next() orelse return error.BadLayerName; // layers
            if (std.mem.eql(u8, layer_or_other, "layers")) {
                // parse layer name
                const layer_index_str = name_segment_iter.next() orelse return error.BadLayerName;
                const layer_desc = name_segment_iter.rest();

                const layer_index = try std.fmt.parseInt(usize, layer_index_str, 10);
                const layer = &model.layers[layer_index];

                if (std.mem.eql(u8, layer_desc, "input_layernorm.weight")) {
                    tensor = &layer.input_layernorm_weight;
                } else if (std.mem.eql(u8, layer_desc, "input_layernorm.bias")) {
                    tensor = &layer.input_layernorm_bias;
                } else if (std.mem.eql(u8, layer_desc, "post_attention_layernorm.weight")) {
                    tensor = &layer.post_attention_layernorm_weight;
                } else if (std.mem.eql(u8, layer_desc, "post_attention_layernorm.bias")) {
                    tensor = &layer.post_attention_layernorm_bias;
                } else if (std.mem.eql(u8, layer_desc, "attention.query.weight")) {
                    tensor = &layer.c_attn_q_proj_w;
                } else if (std.mem.eql(u8, layer_desc, "attention.query.bias")) {
                    tensor = &layer.c_attn_q_proj_bias;
                } else if (std.mem.eql(u8, layer_desc, "attention.key.weight")) {
                    tensor = &layer.c_attn_k_proj_w;
                } else if (std.mem.eql(u8, layer_desc, "attention.key.bias")) {
                    tensor = &layer.c_attn_k_proj_bias;
                } else if (std.mem.eql(u8, layer_desc, "attention.value.weight")) {
                    tensor = &layer.c_attn_v_proj_w;
                } else if (std.mem.eql(u8, layer_desc, "attention.value.bias")) {
                    tensor = &layer.c_attn_v_proj_bias;
                } else if (std.mem.eql(u8, layer_desc, "attention.dense.weight")) {
                    tensor = &layer.c_attn_proj_w;
                } else if (std.mem.eql(u8, layer_desc, "attention.dense.bias")) {
                    tensor = &layer.c_attn_proj_bias;
                } else if (std.mem.eql(u8, layer_desc, "mlp.dense_h_to_4h.weight")) {
                    tensor = &layer.c_mlp_fc_w;
                } else if (std.mem.eql(u8, layer_desc, "mlp.dense_h_to_4h.bias")) {
                    tensor = &layer.c_mlp_fc_b;
                } else if (std.mem.eql(u8, layer_desc, "mlp.dense_4h_to_h.weight")) {
                    tensor = &layer.c_mlp_proj_w_trans;
                } else if (std.mem.eql(u8, layer_desc, "mlp.dense_4h_to_h.bias")) {
                    tensor = &layer.c_mlp_proj_b;
                }
            } else if (std.mem.eql(u8, layer_or_other, "embed_in")) {
                const tensor_type_str = name_segment_iter.rest();
                if (std.mem.eql(u8, tensor_type_str, "weight")) {
                    tensor = &model.wte;
                }
            } else if (std.mem.eql(u8, layer_or_other, "final_layer_norm")) {
                const tensor_type_str = name_segment_iter.rest();
                if (std.mem.eql(u8, tensor_type_str, "weight")) {
                    tensor = &model.ln_f_g;
                } else if (std.mem.eql(u8, tensor_type_str, "bias")) {
                    tensor = &model.ln_f_b;
                }
            } else if (std.mem.eql(u8, name_buffer.items, "embed_out.weight")) {
                tensor = &model.lmh_g;
            }

            if (tensor == null) {
                std.log.warn("{s}: unknown tensor \"{}\"", .{ @src().fn_name, std.zig.fmtEscapes(name_buffer.items) });
                return error.BadTensorName;
            }

            if (c.ggml_nelements(tensor.?.*) != nelements) {
                std.log.warn("{s}: tensor \"{}\" has wrong size in model file, expected {d}, got {d}", .{ @src().fn_name, std.zig.fmtEscapes(name_buffer.items), c.ggml_nelements(tensor.?.*), nelements });
                return error.InvalidTensorSize;
            }

            if (tensor.?.*.ne[0] != shape[0] or tensor.?.*.ne[1] != shape[1]) {
                std.log.warn(" {s}: tensor \"{}\" has wrong shape in model file, expected {any}, got {any}", .{ @src().fn_name, std.zig.fmtEscapes(name_buffer.items), tensor.?.*.ne, shape });
                return error.InvalidTensorShape;
            }

            const bpe_size = switch (ftype) {
                0 => c.ggml_type_size(c.GGML_TYPE_F32),
                1 => c.ggml_type_size(c.GGML_TYPE_F16),
                2 => c.ggml_type_size(c.GGML_TYPE_Q4_0),
                3 => c.ggml_type_size(c.GGML_TYPE_Q4_1),
                else => |f| {
                    std.log.warn(" {s}: unknown ftype {} in model model.file", .{ @src().fn_name, f });
                    return error.UnknownFType;
                },
            };

            if ((nelements * bpe_size) / @intCast(usize, c.ggml_blck_size(tensor.?.*.type)) != c.ggml_nbytes(tensor.?.*)) {
                std.log.warn(" {s}: tensor \"{}\" has wrong size in model model.file, expected {d}, got {d}", .{ @src().fn_name, std.zig.fmtEscapes(name_buffer.items), c.ggml_nbytes(tensor.?.*), nelements * bpe_size });
                return error.InvalidModelSize;
            }

            const offset = try model.file.getPos();
            const tensor_byte_size = c.ggml_nbytes(tensor.?.*);

            tensor.?.*.data = model.file_buf[offset..][0..tensor_byte_size].ptr;

            try model.file.seekBy(@intCast(i64, tensor_byte_size));
            // try model.file.reader().readNoEof(@ptrCast([*]u8, tensor.?.*.data)[0..]);

            total_size += c.ggml_nbytes(tensor.?.*);
            n_tensors += 1;
        }

        std.log.info("{s}: model size = {d:8.2} MB / num_tensors = {}", .{ @src().fn_name, @intToFloat(f32, total_size) / 1024.0 / 1024.0, n_tensors });
    }

    return model;
}

pub fn deinit(this: *@This()) void {
    this.allocator.free(this.layers);
    this.allocator.free(this.ggml_mem_buffer);
    this.tokens.deinit(this.allocator);
    this.tokens_arena.deinit();
}

const std = @import("std");
const c = @import("./c.zig");
