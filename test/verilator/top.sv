module top;

    import "DPI-C" function void init_python_env();
    import "DPI-C" function void deinit_python_env();
    
    // Due to the definition from SV-DPI docs, the appended square bracket pair is required,
    // in despite of the zero dimension data, e.g. `bit[N:0] val` should be `bit[N:0] val[]`,
    // or the compiler will parse it as `svBitVecVal` instead of `svOpenArrayHandle`.

    import "DPI-C" function void array_handle1(
        string filename, string funcname, 
        output bit[15:0] array_out[5:0][1:0][],
        input bit[23:0] array_in[5:0][1:0][]);

    import "DPI-C" function void array_handle2(
        string filename, string funcname,
        output bit[15:0] array_out[1:0][5:0][],
        input bit[23:0] array_in1[5:0][1:0][],
        input bit[23:0] array_in2[1:0][5:0][],
        input bit[31:0] in1_cols,
        input bit[31:0] in1_rows,
        input bit[31:0] in2_cols,
        input bit[31:0] in2_rows,
        input bit[31:0] out_cols,
        input bit[31:0] out_rows);

    import "DPI-C" function void array_handle_1d2d(
        string filename, string funcname,
        output bit[15:0] array_out[5:0][],
        input bit[23:0] array_in1[5:0][],
        input bit[23:0] array_in2[5:0][5:0][]);

    import "DPI-C" function void array_handle_1d2d_model(
        string filename, string funcname, string model_type,
        output bit signed [15:0] array_out[5:0][],
        input bit signed [23:0] array_in1[5:0][],
        input bit signed [23:0] array_in2[5:0][5:0][]);

    initial begin
        bit [15:0] ad_value1 [5:0][1:0];
        bit [15:0] ad_value2 [1:0][5:0];
        
        bit [23:0] da_value1 [5:0][1:0];
        bit [23:0] da_value2 [1:0][5:0];

        bit signed [15:0] ad_value_mvm [5:0];
        bit signed [23:0] da_value_mvm1 [5:0];
        bit signed [23:0] da_value_mvm2 [5:0][5:0];

        foreach(da_value1[i, j])
        begin
            da_value1[i][j] = 24'(i) + 24'(j << 8);
        end

        foreach(da_value2[i, j])
        begin
            da_value2[i][j] = 24'(i << 12) + 24'(j << 8);
        end

        foreach(da_value_mvm1[i])
        begin
            da_value_mvm1[i] = 24'(i);
        end

        // Initialize with mean=0 values: (2*i-5)*10 + (2*j-5)*5
        // For i,j in [0,5]: 2*i-5 gives [-5,-3,-1,1,3,5], sum=0
        foreach(da_value_mvm2[i, j])
        begin
            da_value_mvm2[i][j] = 24'(signed'((2*i - 5) * 10 + (2*j - 5) * 5));
        end

        init_python_env();

        array_handle1("test", "ops_single", ad_value1, da_value1);
            

        foreach(ad_value1[i, j])
        begin
            $display("[SV] ad_value1[%d][%d] = %d", i, j, ad_value1[i][j]);
        end

        // da_value1: [5:0][1:0] = 6 rows, 2 cols
        // da_value2: [1:0][5:0] = 2 rows, 6 cols
        // ad_value2: [1:0][5:0] = 2 rows, 6 cols (output)
        array_handle2("ideal_numerical", "ops_dual", ad_value2, da_value1, da_value2,
            32'd2, 32'd6,   // in1_cols=2, in1_rows=6
            32'd6, 32'd2,   // in2_cols=6, in2_rows=2
            32'd6, 32'd2);  // out_cols=6, out_rows=2
        foreach(ad_value2[i, j])
        begin
            $display("[SV] ad_value2[%d][%d] = %d", i, j, ad_value2[i][j]);
        end

        // Test with different photonic models
        // Available models: "ideal", "noisy", "quantized", "mzi_nonlinear", "all_effects"
        array_handle_1d2d_model("photonic_models", "mvm", "ideal", ad_value_mvm, da_value_mvm1, da_value_mvm2);
        $display("[SV] === MVM Results (model: ideal) ===");
        foreach(ad_value_mvm[i])
        begin
            $display("[SV] ad_value_mvm[%d] = %d", i, $signed(ad_value_mvm[i]));
        end

        deinit_python_env();
        $finish;
    end

endmodule
