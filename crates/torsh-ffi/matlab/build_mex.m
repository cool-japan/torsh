function build_mex()
    % BUILD_MEX Build the ToRSh MEX interface
    %
    % This script compiles the Rust-based ToRSh FFI into a MATLAB MEX file.
    % It requires the Rust toolchain and MATLAB MEX compiler to be installed.
    
    fprintf('Building ToRSh MATLAB MEX interface...\n');
    
    % Get current directory
    current_dir = pwd;
    
    % Change to the parent directory (torsh-ffi root)
    cd('..');
    
    try
        % Build the Rust library first
        fprintf('Building Rust library...\n');
        [status, result] = system('cargo build --release --features matlab');
        
        if status ~= 0
            error('Failed to build Rust library:\n%s', result);
        end
        
        fprintf('Rust library built successfully.\n');
        
        % Get the path to the compiled library
        if ispc
            lib_name = 'torsh_ffi.dll';
            lib_path = fullfile('target', 'release', lib_name);
        elseif ismac
            lib_name = 'libtorsh_ffi.dylib';
            lib_path = fullfile('target', 'release', lib_name);
        else
            lib_name = 'libtorsh_ffi.so';
            lib_path = fullfile('target', 'release', lib_name);
        end
        
        if ~exist(lib_path, 'file')
            error('Compiled library not found at: %s', lib_path);
        end
        
        % Change back to MATLAB directory
        cd('matlab');
        
        % Create MEX wrapper
        fprintf('Creating MEX wrapper...\n');
        create_mex_wrapper();
        
        % Compile MEX file
        fprintf('Compiling MEX file...\n');
        mex_args = {
            'torsh_mex.c',
            ['-L' fullfile('..', 'target', 'release')],
            '-ltorsh_ffi',
            '-output', 'torsh_mex'
        };
        
        % Add platform-specific flags
        if ispc
            mex_args{end+1} = '-DWIN32';
        elseif ismac
            mex_args{end+1} = '-DMACOS';
        else
            mex_args{end+1} = '-DLINUX';
        end
        
        mex(mex_args{:});
        
        fprintf('MEX file compiled successfully.\n');
        
        % Test the MEX file
        fprintf('Testing MEX interface...\n');
        test_mex();
        
        fprintf('ToRSh MATLAB interface is ready!\n');
        fprintf('Use TorshTensor class for tensor operations.\n');
        
    catch ME
        cd(current_dir);
        rethrow(ME);
    end
    
    cd(current_dir);
end

function create_mex_wrapper()
    % Create a C wrapper for the MEX interface
    
    wrapper_code = [
        '#include "mex.h"\n'
        '#include <string.h>\n'
        '\n'
        '// External declarations for Rust functions\n'
        'extern void* matlab_to_torsh_tensor(const mxArray* mx_array);\n'
        'extern mxArray* torsh_tensor_to_matlab(void* tensor);\n'
        'extern mxArray* matlab_tensor_add(const mxArray* lhs, const mxArray* rhs);\n'
        'extern mxArray* matlab_tensor_mul(const mxArray* lhs, const mxArray* rhs);\n'
        'extern mxArray* matlab_tensor_matmul(const mxArray* lhs, const mxArray* rhs);\n'
        'extern mxArray* matlab_tensor_relu(const mxArray* input);\n'
        'extern mxArray* matlab_tensor_zeros(const mxArray* dims);\n'
        'extern mxArray* matlab_tensor_ones(const mxArray* dims);\n'
        '\n'
        'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\n'
        '    if (nrhs < 1) {\n'
        '        mexErrMsgIdAndTxt("ToRSh:InvalidInput", "At least one input required");\n'
        '        return;\n'
        '    }\n'
        '\n'
        '    if (!mxIsChar(prhs[0])) {\n'
        '        mexErrMsgIdAndTxt("ToRSh:InvalidInput", "First argument must be a string");\n'
        '        return;\n'
        '    }\n'
        '\n'
        '    char *command = mxArrayToString(prhs[0]);\n'
        '    if (command == NULL) {\n'
        '        mexErrMsgIdAndTxt("ToRSh:InvalidInput", "Could not convert command to string");\n'
        '        return;\n'
        '    }\n'
        '\n'
        '    if (strcmp(command, "add") == 0) {\n'
        '        if (nrhs != 3 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "add requires 2 inputs and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_add(prhs[1], prhs[2]);\n'
        '    } else if (strcmp(command, "mul") == 0) {\n'
        '        if (nrhs != 3 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "mul requires 2 inputs and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_mul(prhs[1], prhs[2]);\n'
        '    } else if (strcmp(command, "matmul") == 0) {\n'
        '        if (nrhs != 3 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "matmul requires 2 inputs and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_matmul(prhs[1], prhs[2]);\n'
        '    } else if (strcmp(command, "relu") == 0) {\n'
        '        if (nrhs != 2 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "relu requires 1 input and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_relu(prhs[1]);\n'
        '    } else if (strcmp(command, "zeros") == 0) {\n'
        '        if (nrhs != 2 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "zeros requires 1 input and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_zeros(prhs[1]);\n'
        '    } else if (strcmp(command, "ones") == 0) {\n'
        '        if (nrhs != 2 || nlhs != 1) {\n'
        '            mexErrMsgIdAndTxt("ToRSh:InvalidArgs", "ones requires 1 input and 1 output");\n'
        '        }\n'
        '        plhs[0] = matlab_tensor_ones(prhs[1]);\n'
        '    } else {\n'
        '        mexErrMsgIdAndTxt("ToRSh:UnknownCommand", "Unknown command");\n'
        '    }\n'
        '\n'
        '    mxFree(command);\n'
        '}\n'
    ];
    
    % Write the wrapper code to file
    fid = fopen('torsh_mex.c', 'w');
    if fid == -1
        error('Could not create MEX wrapper file');
    end
    
    fprintf(fid, '%s', wrapper_code);
    fclose(fid);
end

function test_mex()
    % Test the MEX interface
    
    try
        % Test tensor creation
        A = TorshTensor([1, 2; 3, 4]);
        B = TorshTensor([5, 6; 7, 8]);
        
        % Test operations
        C = A + B;
        D = A * B;
        E = A.relu();
        
        fprintf('Basic operations test passed.\n');
        
        % Test static methods
        Z = TorshTensor.zeros(3, 3);
        O = TorshTensor.ones(2, 4);
        
        fprintf('Static methods test passed.\n');
        
    catch ME
        warning('MEX test failed: %s', ME.message);
    end
end