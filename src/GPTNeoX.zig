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

memory_k_buf: []u8,
memory_v_buf: []u8,

//
ctx: *c.ggml_context,

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

    model.file_buf = try std.os.mmap(null, file_len, std.os.PROT.READ, std.os.MAP.SHARED, model.file.handle, 0);
    errdefer std.os.munmap(model.file_buf);

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
    for (0..model.hparams.n_vocab) |_| {
        const len = try model.file.reader().readIntLittle(u32);

        const offset = try model.file.getPos();
        const word = model.file_buf[offset..][0..len];
        try model.file.seekBy(len);

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
    const n_mem = model.hparams.n_layer * model.hparams.n_ctx;
    const n_elements = model.hparams.n_embd * n_mem;

    model.memory_k = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, n_elements);
    model.memory_v = c.ggml_new_tensor_1d(model.ctx, c.GGML_TYPE_F32, n_elements);

    const memory_size = c.ggml_nbytes(model.memory_k) + c.ggml_nbytes(model.memory_v);

    model.memory_k_buf = try allocator.alloc(u8, memory_size);
    errdefer allocator.free(model.memory_k_buf);

    model.memory_v_buf = try allocator.alloc(u8, memory_size);
    errdefer allocator.free(model.memory_v_buf);

    model.memory_k.*.data = model.memory_k_buf.ptr;
    model.memory_v.*.data = model.memory_v_buf.ptr;

    std.log.info("{s}: memory size = {d:8.2} MB, n_mem = {d}", .{ @src().fn_name, @intToFloat(f32, memory_size) / (1024 * 1024), n_mem });

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
    std.os.munmap(this.file_buf);
    this.file.close();
}

const EvalOptions = struct {
    n_threads: usize = 1,
    n_past: usize = 10,
};

pub fn eval(this: @This(), allocator: std.mem.Allocator, input_embeddings: []const usize, options: EvalOptions) ![]f32 {
    const buf = try allocator.alloc(u8, 100 * 1024 * 1024);
    defer allocator.free(buf);
    var params = c.ggml_init_params{
        .mem_buffer = buf.ptr,
        .mem_size = buf.len,
        .no_alloc = false,
    };

    const ctx = c.ggml_init(params);
    defer c.ggml_free(ctx);

    var compute_graph = std.mem.zeroInit(c.ggml_cgraph, .{
        .n_threads = 1,
    });

    const embd = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @intCast(i64, input_embeddings.len));
    std.mem.copy(usize, @ptrCast([*]usize, @alignCast(@alignOf(usize), embd.*.data))[0..input_embeddings.len], input_embeddings);

    var input_layer = c.ggml_get_rows(ctx, this.wte, embd);
    for (this.layers, 0..) |layer, il| {
        var cur = c.ggml_norm(ctx, input_layer);
        cur = c.ggml_add(
            ctx,
            c.ggml_mul(
                ctx,
                c.ggml_repeat(ctx, layer.input_layernorm_weight, cur),
                cur,
            ),
            c.ggml_repeat(ctx, layer.input_layernorm_bias, cur),
        );

        var q_cur = c.ggml_mul_mat(ctx, layer.c_attn_q_proj_w, cur);
        var k_cur = c.ggml_mul_mat(ctx, layer.c_attn_k_proj_w, cur);
        var v_cur = c.ggml_mul_mat(ctx, layer.c_attn_v_proj_w, cur);

        q_cur = c.ggml_add(ctx, q_cur, c.ggml_repeat(ctx, layer.c_attn_q_proj_bias, q_cur));
        k_cur = c.ggml_add(ctx, k_cur, c.ggml_repeat(ctx, layer.c_attn_k_proj_bias, k_cur));
        v_cur = c.ggml_add(ctx, v_cur, c.ggml_repeat(ctx, layer.c_attn_v_proj_bias, v_cur));

        if (input_embeddings.len >= 1) {
            const k = c.ggml_view_1d(ctx, this.memory_k, @intCast(i64, input_embeddings.len * this.hparams.n_embd), (c.ggml_element_size(this.memory_k) * this.hparams.n_embd) * (il * this.hparams.n_ctx + options.n_past));
            const v = c.ggml_view_1d(ctx, this.memory_v, @intCast(i64, input_embeddings.len * this.hparams.n_embd), (c.ggml_element_size(this.memory_v) * this.hparams.n_embd) * (il * this.hparams.n_ctx + options.n_past));

            c.ggml_build_forward_expand(&compute_graph, c.ggml_cpy(ctx, k_cur, k));
            c.ggml_build_forward_expand(&compute_graph, c.ggml_cpy(ctx, v_cur, v));
        }

        var q = c.ggml_permute(ctx, c.ggml_rope(
            ctx,
            c.ggml_cpy(ctx, q_cur, c.ggml_new_tensor_3d(
                ctx,
                c.GGML_TYPE_F32,
                this.hparams.n_embd / this.hparams.n_head,
                this.hparams.n_head,
                @intCast(i64, input_embeddings.len),
            )),
            @intCast(c_int, options.n_past),
            @intCast(c_int, this.hparams.n_rot),
            0,
        ), 0, 2, 1, 3);

        var k = c.ggml_permute(ctx, c.ggml_rope(
            ctx, // "change Qcur" in line 2270.)
            c.ggml_reshape_3d(
                ctx,
                c.ggml_view_1d(
                    ctx,
                    this.memory_k,
                    @intCast(i64, (options.n_past + input_embeddings.len) * this.hparams.n_embd),
                    il * this.hparams.n_ctx * c.ggml_element_size(this.memory_k) * this.hparams.n_embd,
                ),
                this.hparams.n_embd / this.hparams.n_head,
                this.hparams.n_head,
                @intCast(i64, options.n_past + input_embeddings.len),
            ),
            @intCast(c_int, options.n_past),
            @intCast(c_int, this.hparams.n_rot),
            1,
        ), 0, 2, 1, 3);
        var kq = c.ggml_mul_mat(ctx, k, q);

        var kq_scaled = c.ggml_scale(
            ctx,
            kq,
            c.ggml_new_f32(
                ctx,
                1.0 / @sqrt(@intToFloat(f32, this.hparams.n_embd) / @intToFloat(f32, this.hparams.n_head)),
            ),
        );

        var kq_masked = c.ggml_diag_mask_inf(ctx, kq_scaled, @intCast(c_int, options.n_past));
        const kq_soft_max = c.ggml_soft_max(ctx, kq_masked);
        const v_trans = c.ggml_dup_tensor(ctx, c.ggml_permute(
            ctx,
            c.ggml_reshape_3d(
                ctx,
                c.ggml_view_1d(
                    ctx,
                    this.memory_v,
                    @intCast(i64, (options.n_past + input_embeddings.len) * this.hparams.n_embd),
                    il * this.hparams.n_ctx * c.ggml_element_size(this.memory_v) * this.hparams.n_embd,
                ),
                this.hparams.n_embd / this.hparams.n_head,
                this.hparams.n_head,
                @intCast(i64, options.n_past + input_embeddings.len),
            ),
            1,
            2,
            0,
            3,
        ));
        const kqv = c.ggml_mul_mat(ctx, v_trans, kq_soft_max);
        const kqv_merged = c.ggml_permute(ctx, kqv, 0, 2, 1, 3);

        cur = c.ggml_cpy(ctx, kqv_merged, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, this.hparams.n_embd, @intCast(i64, input_embeddings.len)));

        // projection (first weight)
        cur = c.ggml_mul_mat(ctx, layer.c_attn_proj_w, cur);

        // projection (then bias)
        cur = c.ggml_add(ctx, c.ggml_repeat(ctx, layer.c_attn_proj_bias, cur), cur);

        var inpFF: *c.ggml_tensor = undefined;

        if (this.hparams.use_parallel_residual == 0) {
            std.debug.print("use_parallel_residual == 0\n", .{});
            // This takes the self-attention residual output as input to Feedforward
            inpFF = c.ggml_add(ctx, cur, input_layer);

            // post attention layer norm
            {
                inpFF = c.ggml_norm(ctx, inpFF);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = c.ggml_add(ctx, c.ggml_mul(ctx, c.ggml_repeat(ctx, layer.post_attention_layernorm_weight, inpFF), inpFF), c.ggml_repeat(ctx, layer.post_attention_layernorm_bias, inpFF));
            }

            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = c.ggml_mul_mat(ctx, layer.c_mlp_fc_w, inpFF);

                inpFF = c.ggml_add(ctx, c.ggml_repeat(ctx, layer.c_mlp_fc_b, inpFF), inpFF);

                inpFF = c.ggml_gelu(ctx, inpFF);

                inpFF = c.ggml_mul_mat(ctx, layer.c_mlp_proj_w_trans, inpFF);

                inpFF = c.ggml_add(ctx, c.ggml_repeat(ctx, layer.c_mlp_proj_b, inpFF), inpFF);
            }

            // input_layer = inpFF + input_layer
            input_layer = c.ggml_add(ctx, inpFF, input_layer);
        } else if (this.hparams.use_parallel_residual == 1) {
            // printf("use_parallel_residual == 1\n");
            // This is independent of the self-attention result, so it could be done in parallel to the self-attention

            // post attention layer norm
            {
                inpFF = c.ggml_norm(ctx, input_layer);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = c.ggml_add(ctx, c.ggml_mul(ctx, c.ggml_repeat(ctx, layer.post_attention_layernorm_weight, inpFF), inpFF), c.ggml_repeat(ctx, layer.post_attention_layernorm_bias, inpFF));
            }

            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = c.ggml_mul_mat(ctx, layer.c_mlp_fc_w, inpFF);

                inpFF = c.ggml_add(ctx, c.ggml_repeat(ctx, layer.c_mlp_fc_b, inpFF), inpFF);

                // GELU activation
                inpFF = c.ggml_gelu(ctx, inpFF);

                // projection
                // inpFF = proj_w*inpFF + proj_b
                inpFF = c.ggml_mul_mat(ctx, layer.c_mlp_proj_w_trans, inpFF);

                inpFF = c.ggml_add(ctx, c.ggml_repeat(ctx, layer.c_mlp_proj_b, inpFF), inpFF);
            }

            // r = r + inpFF + cur
            inpFF = c.ggml_add(ctx, cur, inpFF);
            input_layer = c.ggml_add(ctx, input_layer, inpFF);
        } else {
            std.debug.panic("use_parallel_residual == {}\n", .{this.hparams.use_parallel_residual});
        }
    }

    // norm
    {
        input_layer = c.ggml_norm(ctx, input_layer);

        // input_layer = ln_f_g*input_layer + ln_f_b
        input_layer = c.ggml_add(ctx, c.ggml_mul(ctx, c.ggml_repeat(ctx, this.ln_f_g, input_layer), input_layer), c.ggml_repeat(ctx, this.ln_f_b, input_layer));
    }

    // lm_head
    {
        input_layer = c.ggml_mul_mat(ctx, this.lmh_g, input_layer);
    }

    // run the computation
    c.ggml_build_forward_expand(&compute_graph, input_layer);
    c.ggml_graph_compute(ctx, &compute_graph);

    return &.{};
}

const std = @import("std");
const c = @import("./c.zig");
