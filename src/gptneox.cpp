
// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-NeoX model requires about 16MB of memory per input token.
//
bool gptneox_eval(
        const gptneox_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

     embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N): *c.ggml_tensor ;
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // wte
     inpL = ggml_get_rows(ctx0, model.wte, embd): *c.ggml_tensor ;

    // for (int il = 0; il < 1; ++il) {
    for (int il = 0; il < n_layer; ++il) {
         cur: *c.ggml_tensor ;

        // input norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = input_layernorm_weight*cur + input_layernorm_bias
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].input_layernorm_weight, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].input_layernorm_bias, cur));

        }

        // self-attention
        {
            // Weight
             Qcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_q_proj_w, cur): *c.ggml_tensor ;
             Kcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_k_proj_w, cur): *c.ggml_tensor ;
             Vcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_v_proj_w, cur): *c.ggml_tensor ;

            // Add bias
            Qcur = ggml_add(ctx0, Qcur, ggml_repeat(ctx0, model.layers[il].c_attn_q_proj_bias, Qcur));
            Kcur = ggml_add(ctx0, Kcur, ggml_repeat(ctx0, model.layers[il].c_attn_k_proj_bias, Kcur));
            Vcur = ggml_add(ctx0, Vcur, ggml_repeat(ctx0, model.layers[il].c_attn_v_proj_bias, Vcur));

            // // // // cur = ggml_add(ctx0, cur, Qcur);
            // // // // cur = ggml_add(ctx0, cur, Kcur);
            // // // // cur = ggml_add(ctx0, cur, Vcur);

            // store key and value to memory
            if (N >= 1) {
                 k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past)): *c.ggml_tensor ;
                 v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past)): *c.ggml_tensor ;

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
             Q =
                ggml_permute(ctx0,
                        ggml_gptneox_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3): *c.ggml_tensor ;

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
             K =
                ggml_permute(ctx0,
                        ggml_gptneox_rope(ctx0,// "change Qcur" in line 2270.)
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3): *c.ggml_tensor ;

            // K * Q
             KQ = ggml_mul_mat(ctx0, K, Q): *c.ggml_tensor ;

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
             KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        ): *c.ggml_tensor ;

            // KQ_masked = mask_past(KQ_scaled)
             KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past): *c.ggml_tensor ;

            // KQ = soft_max(KQ_masked)
             KQ_soft_max = ggml_soft_max(ctx0, KQ_masked): *c.ggml_tensor ;

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
             V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3): *c.ggml_tensor ;

            // KQV = transpose(V) * KQ_soft_max
             KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max): *c.ggml_tensor ;

            // KQV_merged = KQV.permute(0, 2, 1, 3)
             KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3): *c.ggml_tensor ;

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_proj_w, cur);

            // projection (then bias)
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_attn_proj_bias, cur), cur);
        }

         inpFF: *c.ggml_tensor ;

        if (hparams.use_parallel_residual == 0) {
            printf("use_parallel_residual == 0\n");
            // This takes the self-attention residual output as input to Feedforward
            inpFF = ggml_add(ctx0, cur, inpL);

            // post attention layer norm
            {
                inpFF = ggml_norm(ctx0, inpFF);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, inpFF),
                        inpFF),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, inpFF));
            }

            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, inpFF), inpFF);

                inpFF = ggml_gelu(ctx0, inpFF);

                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, inpFF), inpFF);
            }

            // inpL = inpFF + inpL
            inpL = ggml_add(ctx0, inpFF, inpL);

        } else if (hparams.use_parallel_residual == 1) {
            // printf("use_parallel_residual == 1\n");
            // This is independent of the self-attention result, so it could be done in parallel to the self-attention

            // post attention layer norm
            {
                inpFF = ggml_norm(ctx0, inpL);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, inpFF),
                        inpFF),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, inpFF));
            }


            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, inpFF), inpFF);

                // GELU activation
                inpFF = ggml_gelu(ctx0, inpFF);

                // projection
                // inpFF = proj_w*inpFF + proj_b
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, inpFF), inpFF);
            }

            // inpL = inpL + inpFF + cur
            inpFF = ggml_add(ctx0, cur, inpFF);
            inpL = ggml_add(ctx0, inpL, inpFF);
        } else {
            printf("use_parallel_residual == %d\n", hparams.use_parallel_residual);
            assert(0);
        }
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main_gptneox(gpt_params params) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
    int64_t t_load_us = 0;

    std::mt19937 rng(params.seed);

    gpt_vocab vocab;
    gptneox_model model;
    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        const int n_ctx = 512; // TODO: set context from user input ??
        if (!gptneox_model_load(params.model, model, vocab, n_ctx)) {  // TODO: set context from user input ??
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::whitespace_tokenize(params.prompt); //TODO: set bos to true?

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    // for (int i = 0; i < (int) embd_inp.size(); i++) {
    //     printf("%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    // }
    printf("\n");
    printf("sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    printf("\n\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gptneox_eval(model, params.n_threads, 0, { 1, 2, 3, 4, 5 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    printf("\n<|BEGIN> ");
    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gptneox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) { // update logits
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            if (params.return_logits) {
                printf("logits: ");
                for (int i = 0; i < n_vocab; i++) {
                    // Upto 8 decimal places
                    printf("%.8f ", logits[i]);
                }
                printf(" <END|>\n");
                // Stdout should flush before returning
                fflush(stdout);
                return 0;
            }

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = sample_top_p_top_k_repeat_penalty(
                        vocab,
                        logits.data() + (logits.size() - n_vocab),
                        last_n_tokens,
                        repeat_penalty,
                        top_k,
                        top_p,
                        temp,
                        rng);

                // // print
                // printf("\ngenerated token: '%s' (%d)\n", vocab.id_to_token[id].c_str(), id);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                if (params.return_logits) {
                    printf("logits: ");
                    for (int i = 0; i < model.hparams.n_vocab; i++) {
                        // Upto 8 decimal places
                        printf("%.8f ", logits[i]);
                    }
                    printf("\n");
                }
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            if (!params.return_logits) {
                printf(" %d ", id);
            }
            // printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 2) {
            break;
        }
    }
    printf(" <END|>\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}

