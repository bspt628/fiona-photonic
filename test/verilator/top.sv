module top;

    // ============================================================
    // Parameters
    // ============================================================
    parameter int ROWS = 6;
    parameter int COLS = 6;
    parameter int VEC_LEN = 6;

    // ============================================================
    // DPI-C Function Imports
    // ============================================================
    import "DPI-C" function void init_python_env();
    import "DPI-C" function void deinit_python_env();

    // Due to the definition from SV-DPI docs, the appended square bracket pair is required,
    // in despite of the zero dimension data, e.g. `bit[N:0] val` should be `bit[N:0] val[]`,
    // or the compiler will parse it as `svBitVecVal` instead of `svOpenArrayHandle`.

    // Handler 1: Single input/output (test/ops_single)
    import "DPI-C" function void array_handle1(
        string filename, string funcname,
        output bit[15:0] array_out[ROWS-1:0][1:0][],
        input bit[15:0] array_in[ROWS-1:0][1:0][]);

    // Handler 2: Dual input with padding (ideal_numerical/ops_dual)
    import "DPI-C" function void array_handle2(
        string filename, string funcname,
        output bit[15:0] array_out[1:0][COLS-1:0][],
        input bit[15:0] array_in1[ROWS-1:0][1:0][],
        input bit[15:0] array_in2[1:0][COLS-1:0][],
        input bit[31:0] in1_cols,
        input bit[31:0] in1_rows,
        input bit[31:0] in2_cols,
        input bit[31:0] in2_rows,
        input bit[31:0] out_cols,
        input bit[31:0] out_rows);

    // Handler 3: 1D vector + 2D matrix (unused, kept for reference)
    import "DPI-C" function void array_handle_1d2d(
        string filename, string funcname,
        output bit[15:0] array_out[VEC_LEN-1:0][],
        input bit[15:0] array_in1[VEC_LEN-1:0][],
        input bit[15:0] array_in2[ROWS-1:0][COLS-1:0][]);

    // Handler 4: 1D+2D with model selection (photonic_models/mvm)
    import "DPI-C" function void array_handle_1d2d_model(
        string filename, string funcname, string model_type,
        output bit signed [15:0] array_out[VEC_LEN-1:0][],
        input bit signed [15:0] array_in1[VEC_LEN-1:0][],
        input bit signed [15:0] array_in2[ROWS-1:0][COLS-1:0][]);

    // Handler 5: 1D+2D with FP32 support (photonic_models/mvm_fp32)
    import "DPI-C" function void array_handle_1d2d_fp32(
        string filename, string funcname, string model_type,
        output bit [31:0] array_out[VEC_LEN-1:0][],
        input bit [31:0] array_in1[VEC_LEN-1:0][],
        input bit [31:0] array_in2[ROWS-1:0][COLS-1:0][]);

    initial begin
        // ============================================================
        // Variables for array_handle1 (test/ops_single)
        // ============================================================
        bit [15:0] h1_da_in [ROWS-1:0][1:0];   // DAC input
        bit [15:0] h1_ad_out [ROWS-1:0][1:0];  // ADC output

        // ============================================================
        // Variables for array_handle2 (ideal_numerical/ops_dual)
        // ============================================================
        bit [15:0] h2_da_in1 [ROWS-1:0][1:0];  // DAC input 1
        bit [15:0] h2_da_in2 [1:0][COLS-1:0];  // DAC input 2
        bit [15:0] h2_ad_out [1:0][COLS-1:0];  // ADC output

        // ============================================================
        // Variables for array_handle_1d2d_model (photonic_models/mvm)
        // ============================================================
        bit signed [15:0] mvm_da_vec [VEC_LEN-1:0];          // DAC input vector
        bit signed [15:0] mvm_da_mat [VEC_LEN-1:0][VEC_LEN-1:0];   // DAC input matrix
        bit signed [15:0] mvm_ad_out [VEC_LEN-1:0];          // ADC output (ideal)
        bit signed [15:0] mvm_ad_out_noisy [VEC_LEN-1:0];    // ADC output (noisy)

        // ============================================================
        // Variables for array_handle_1d2d_fp32 (photonic_models/mvm_fp32)
        // ============================================================
        bit [31:0] fp32_da_vec [VEC_LEN-1:0];                // FP32 input vector
        bit [31:0] fp32_da_mat [VEC_LEN-1:0][VEC_LEN-1:0];   // FP32 input matrix
        bit [31:0] fp32_ad_out [VEC_LEN-1:0];                // FP32 output vector

        // ============================================================
        // Initialize test data for array_handle1
        // ============================================================
        foreach(h1_da_in[i, j])
        begin
            h1_da_in[i][j] = 16'(i) + 16'(j << 8);
        end

        // ============================================================
        // Initialize test data for array_handle2
        // ============================================================
        foreach(h2_da_in1[i, j])
        begin
            h2_da_in1[i][j] = 16'(i) + 16'(j << 8);
        end

        foreach(h2_da_in2[i, j])
        begin
            h2_da_in2[i][j] = 16'(i << 12) + 16'(j << 8);
        end

        // ============================================================
        // Initialize test data for MVM
        // Vector: [0, 1, 2, 3, 4, 5]
        // Matrix: mean=0 values using (2*i-5)*10 + (2*j-5)*5
        // ============================================================
        foreach(mvm_da_vec[i])
        begin
            mvm_da_vec[i] = 16'(i);
        end

        foreach(mvm_da_mat[i, j])
        begin
            // (2*i-5)*10 + (2*j-5)*5 gives values from -75 to +75 with mean=0
            mvm_da_mat[i][j] = 16'(signed'((2*i - 5) * 10 + (2*j - 5) * 5));
        end

        // ============================================================
        // Initialize FP32 test data
        // Vector: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        // IEEE 754 single-precision bit patterns:
        //   0.5  = 0x3F000000    1.5  = 0x3FC00000
        //   2.5  = 0x40200000    3.5  = 0x40600000
        //   4.5  = 0x40900000    5.5  = 0x40B00000
        // ============================================================
        fp32_da_vec[0] = 32'h3F000000;  // 0.5
        fp32_da_vec[1] = 32'h3FC00000;  // 1.5
        fp32_da_vec[2] = 32'h40200000;  // 2.5
        fp32_da_vec[3] = 32'h40600000;  // 3.5
        fp32_da_vec[4] = 32'h40900000;  // 4.5
        fp32_da_vec[5] = 32'h40B00000;  // 5.5

        // Matrix: simple diagonal-dominant values for easy verification
        // Row i: diagonal=1.0, off-diagonal=0.1
        // IEEE 754: 1.0 = 0x3F800000, 0.1 ≈ 0x3DCCCCCD
        foreach(fp32_da_mat[i, j])
        begin
            if (i == j)
                fp32_da_mat[i][j] = 32'h3F800000;  // 1.0
            else
                fp32_da_mat[i][j] = 32'h3DCCCCCD;  // 0.1
        end

        // ============================================================
        // Execute tests
        // ============================================================
        init_python_env();

        // Test 1: array_handle1 (test/ops_single)
        $display("\n[SV] ========== Test: array_handle1 (test/ops_single) ==========");
        array_handle1("test", "ops_single", h1_ad_out, h1_da_in);
        foreach(h1_ad_out[i, j])
        begin
            $display("[SV] h1_ad_out[%0d][%0d] = %0d", i, j, h1_ad_out[i][j]);
        end

        // Test 2: array_handle2 (ideal_numerical/ops_dual)
        $display("\n[SV] ========== Test: array_handle2 (ideal_numerical/ops_dual) ==========");
        array_handle2("ideal_numerical", "ops_dual", h2_ad_out, h2_da_in1, h2_da_in2,
            32'd2, 32'd6,   // in1: cols=2, rows=6
            32'd6, 32'd2,   // in2: cols=6, rows=2
            32'd6, 32'd2);  // out: cols=6, rows=2
        foreach(h2_ad_out[i, j])
        begin
            $display("[SV] h2_ad_out[%0d][%0d] = %0d", i, j, h2_ad_out[i][j]);
        end

        // Test 3: MVM with ideal model
        // Available models: "ideal", "noisy", "quantized", "mzi_nonlinear", "all_effects"
        $display("\n[SV] ========== Test: MVM (model: ideal) ==========");
        array_handle_1d2d_model("photonic_models", "mvm", "ideal",
            mvm_ad_out, mvm_da_vec, mvm_da_mat);
        foreach(mvm_ad_out[i])
        begin
            $display("[SV] mvm_ad_out[%0d] = %0d", i, $signed(mvm_ad_out[i]));
        end

        // Test 4: MVM with noisy model (legacy)
        $display("\n[SV] ========== Test: MVM (model: noisy) ==========");
        array_handle_1d2d_model("photonic_models", "mvm", "noisy",
            mvm_ad_out_noisy, mvm_da_vec, mvm_da_mat);
        foreach(mvm_ad_out_noisy[i])
        begin
            $display("[SV] mvm_ad_out_noisy[%0d] = %0d", i, $signed(mvm_ad_out_noisy[i]));
        end

        // Test 5: MVM with realistic MZI model
        // This model includes: phase error, thermal crosstalk, insertion loss,
        // output crosstalk, detector noise, and DAC/ADC quantization
        $display("\n[SV] ========== Test: MVM (model: mzi_realistic) ==========");
        array_handle_1d2d_model("photonic_models", "mvm", "mzi_realistic",
            mvm_ad_out_noisy, mvm_da_vec, mvm_da_mat);
        foreach(mvm_ad_out_noisy[i])
        begin
            $display("[SV] mvm_ad_out_realistic[%0d] = %0d", i, $signed(mvm_ad_out_noisy[i]));
        end

        // Test 6: FP32 MVM (Transformer-style floating point)
        // Expected output (for diagonal-dominant matrix):
        //   out[i] = vec[i]*1.0 + sum(vec[j]*0.1 for j!=i)
        //   out[0] = 0.5*1.0 + (1.5+2.5+3.5+4.5+5.5)*0.1 = 0.5 + 1.75 = 2.25
        //   out[1] = 1.5*1.0 + (0.5+2.5+3.5+4.5+5.5)*0.1 = 1.5 + 1.65 = 3.15
        //   out[2] = 2.5*1.0 + (0.5+1.5+3.5+4.5+5.5)*0.1 = 2.5 + 1.55 = 4.05
        //   ... etc
        $display("\n[SV] ========== Test: FP32 MVM (model: ideal) ==========");
        $display("[SV] Input vector (FP32): [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]");
        $display("[SV] Weight matrix: diagonal=1.0, off-diagonal=0.1");
        array_handle_1d2d_fp32("photonic_models", "mvm_fp32", "ideal",
            fp32_ad_out, fp32_da_vec, fp32_da_mat);
        // Display output as hex (Python side will show float values)
        foreach(fp32_ad_out[i])
        begin
            $display("[SV] fp32_ad_out[%0d] = 0x%08h", i, fp32_ad_out[i]);
        end

        deinit_python_env();
        $finish;
    end

endmodule
