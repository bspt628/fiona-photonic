#ifndef FIONA_PHOTONIC_ENGINE
#define FIONA_PHOTONIC_ENGINE

// Standard Library Header (always needed)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <functional>
#include <string>

#ifdef USE_PYTHON
// Python Development Header (only when using Python backend)
#include <Python.h>
#include <abstract.h>
#include <boolobject.h>
#include <import.h>
#include <listobject.h>
#include <longobject.h>
#include <object.h>
#include <tupleobject.h>
#include <unicodeobject.h>
#include <dlfcn.h>

// Photonic Model Register Header (Python mode only)
#include "register.h"

extern const PyFileFuncVec pyfilefunc_reg;

typedef std::map<PyFileFunc, PyObject*> PyFuncMap;
PyFuncMap pyfunc_map;

#endif /* USE_PYTHON */

// Eigen Library Header (used for both Python and C++ modes)
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif /* USE_EIGEN */


/********** Constants **********/
#define FIONA_VLEN_MAX 32

#ifdef USE_PYTHON
/********** MVM Batch Buffer for Simulation Optimization (Python Mode) **********/
/*
 * This buffer collects MVM requests and processes them in batches,
 * reducing Python call overhead from ~50ms per call to ~50ms per batch.
 *
 * Design: Option B (Bridge Layer Buffering)
 * - fiona-spikesim (fiona.cc) remains unchanged
 * - Buffering logic is contained within fiona-photonic
 *
 * Environment variables:
 * - FIONA_BATCH_ENABLED: 1 (default) or 0
 * - FIONA_BATCH_SIZE: max batch size, default 256
 * - FIONA_BATCH_DEBUG: 1 to enable debug output
 */

struct MVMRequest {
    int16_t mat[FIONA_VLEN_MAX * FIONA_VLEN_MAX];
    int16_t vec[FIONA_VLEN_MAX];
    size_t vlen;
};

struct MVMResult {
    int16_t* data;
    size_t rows;
    size_t cols;
};

class MVMBatchBuffer {
private:
    std::vector<MVMRequest> requests;
    std::vector<MVMResult> results;
    size_t next_result_idx;
    size_t max_batch_size;
    bool enabled;
    bool debug;

    static bool get_env_bool(const char* name, bool default_val) {
        const char* val = std::getenv(name);
        if (val == nullptr) return default_val;
        return std::string(val) == "1";
    }

    static size_t get_env_size(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (val == nullptr) return default_val;
        return std::stoul(val);
    }

public:
    MVMBatchBuffer() : next_result_idx(0) {
        enabled = get_env_bool("FIONA_BATCH_ENABLED", true);
        max_batch_size = get_env_size("FIONA_BATCH_SIZE", 256);
        debug = get_env_bool("FIONA_BATCH_DEBUG", false);

        if (debug) {
            std::cout << "[BATCH] Initialized: enabled=" << enabled
                      << ", max_size=" << max_batch_size << std::endl;
        }
    }

    bool is_enabled() const { return enabled; }

    // Add a request to the buffer
    size_t add_request(int16_t* mat, int16_t* vec, size_t vlen) {
        MVMRequest req;
        std::memcpy(req.mat, mat, FIONA_VLEN_MAX * FIONA_VLEN_MAX * sizeof(int16_t));
        std::memcpy(req.vec, vec, FIONA_VLEN_MAX * sizeof(int16_t));
        req.vlen = vlen;
        requests.push_back(req);

        if (debug) {
            std::cout << "[BATCH] Added request #" << (requests.size() - 1)
                      << ", buffer size: " << requests.size() << std::endl;
        }

        return requests.size() - 1;
    }

    // Check if buffer should be flushed
    bool should_flush() const {
        return requests.size() >= max_batch_size;
    }

    // Get current buffer size
    size_t size() const { return requests.size(); }

    // Flush: call Python mvm_batched and store results
    void flush(PyFuncMap& pyfunc_map) {
        if (requests.empty()) return;

        if (debug) {
            std::cout << "[BATCH] Flushing " << requests.size() << " requests" << std::endl;
        }

        // Build Python list of dicts for batch
        PyObject* batch_list = PyList_New(requests.size());

        for (size_t i = 0; i < requests.size(); i++) {
            PyObject* item = PyDict_New();

            // 'mat': flattened matrix as list
            PyObject* mat_list = PyList_New(FIONA_VLEN_MAX * FIONA_VLEN_MAX);
            for (size_t j = 0; j < FIONA_VLEN_MAX * FIONA_VLEN_MAX; j++) {
                PyList_SetItem(mat_list, j, PyLong_FromLong(requests[i].mat[j]));
            }

            // 'vec': vector as list
            PyObject* vec_list = PyList_New(FIONA_VLEN_MAX);
            for (size_t j = 0; j < FIONA_VLEN_MAX; j++) {
                PyList_SetItem(vec_list, j, PyLong_FromLong(requests[i].vec[j]));
            }

            PyDict_SetItemString(item, "mat", mat_list);
            PyDict_SetItemString(item, "vec", vec_list);
            PyDict_SetItemString(item, "vlen", PyLong_FromLong(requests[i].vlen));
            PyDict_SetItemString(item, "rd", PyLong_FromLong(i));

            PyList_SetItem(batch_list, i, item);

            Py_DecRef(mat_list);
            Py_DecRef(vec_list);
        }

        // Call Python mvm_batched
        PyObject* callable = pyfunc_map[PyFileFunc("photonic_models", "mvm_batched")];
        if (callable && PyCallable_Check(callable)) {
            PyObject* args = PyTuple_Pack(1, batch_list);
            PyObject* ret_obj = PyObject_CallObject(callable, args);

            if (ret_obj == nullptr) {
                PyErr_Print();
                std::cerr << "[BATCH] Error calling mvm_batched" << std::endl;
                Py_DecRef(args);
                Py_DecRef(batch_list);
                return;
            }

            // Parse results: list of dicts with 'result' and 'rd'
            size_t num_results = PyList_Size(ret_obj);
            results.clear();
            results.resize(num_results);

            for (size_t i = 0; i < num_results; i++) {
                PyObject* result_item = PyList_GetItem(ret_obj, i);
                PyObject* result_list = PyDict_GetItemString(result_item, "result");
                size_t result_size = PyList_Size(result_list);

                int16_t* result_data = new int16_t[result_size];
                for (size_t j = 0; j < result_size; j++) {
                    result_data[j] = (int16_t)PyLong_AsLong(PyList_GetItem(result_list, j));
                }

                results[i].data = result_data;
                results[i].rows = result_size;
                results[i].cols = 1;
            }

            Py_DecRef(ret_obj);
            Py_DecRef(args);
        }

        Py_DecRef(batch_list);

        // Reset for next batch
        requests.clear();
        next_result_idx = 0;

        if (debug) {
            std::cout << "[BATCH] Flush complete, " << results.size() << " results ready" << std::endl;
        }
    }

    // Get result for a specific request
    // If not yet flushed, triggers flush first
    int16_t* get_result(size_t request_id, size_t& rows, size_t& cols, PyFuncMap& pyfunc_map) {
        // If results not ready, flush now
        if (results.empty() || request_id >= results.size()) {
            flush(pyfunc_map);
        }

        if (request_id < results.size()) {
            rows = results[request_id].rows;
            cols = results[request_id].cols;
            return results[request_id].data;
        }

        // Fallback: should not happen
        std::cerr << "[BATCH] Warning: result not found for request " << request_id << std::endl;
        rows = 0;
        cols = 0;
        return nullptr;
    }

    // Clear all cached results (call when done with current batch)
    void clear_results() {
        for (auto& r : results) {
            if (r.data) {
                delete[] r.data;
                r.data = nullptr;
            }
        }
        results.clear();
        next_result_idx = 0;
    }

    ~MVMBatchBuffer() {
        clear_results();
    }
};

// Global batch buffer instance
static MVMBatchBuffer g_mvm_batch_buffer;

// Flag to track if we're in batch mode (set after first MVM in a batch)
static bool g_batch_has_pending = false;

/********** External Interface for fiona.cc **********/
// These functions are called from fiona-spikesim to control batch processing

// Flush all pending MVM requests
inline void batch_flush() {
    if (g_mvm_batch_buffer.size() > 0) {
        g_mvm_batch_buffer.flush(pyfunc_map);
        g_batch_has_pending = false;
    }
}

// Check if batch processing is enabled
inline bool batch_is_enabled() {
    return g_mvm_batch_buffer.is_enabled();
}

// Get number of pending requests
inline size_t batch_pending_count() {
    return g_mvm_batch_buffer.size();
}

// Add MVM request to batch and return result
// This is the main entry point for batched MVM from fiona.cc
inline int16_t* batch_mvm(int16_t* mat, int16_t* vec, size_t vlen, size_t& out_rows) {
    size_t req_id = g_mvm_batch_buffer.add_request(mat, vec, vlen);
    g_batch_has_pending = true;

    // Check if we should auto-flush (batch full)
    if (g_mvm_batch_buffer.should_flush()) {
        g_mvm_batch_buffer.flush(pyfunc_map);
        g_batch_has_pending = false;
    }

    // If results are available, return them
    // Otherwise, flush now to get results (for immediate use)
    size_t rows, cols;
    int16_t* result = g_mvm_batch_buffer.get_result(req_id, rows, cols, pyfunc_map);
    out_rows = rows;
    return result;
}


/********** Conversion between EigenMatrix and PointerBuffer **********/
#ifdef USE_EIGEN

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_from_buffer(T *array, size_t rows, size_t cols) {
    return Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(array, rows, cols);
}

template <typename T>
T* buffer_from_matrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, size_t *rows, size_t *cols) {
    *rows = mat.rows();
    *cols = mat.cols();
    
    T buffer[*rows][*cols];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(&buffer[0][0], mat.rows(), mat.cols()) = mat;

    T *array = new T[mat.size()];
    for(auto i = 0; i < *rows; ++i) {
        for(auto j = 0; j < *cols; ++j) {
            auto offset = *cols * i + j;
            array[offset] = buffer[i][j];
        }
    }
    return array;
}

#endif /* USE_EIGEN */

/********** Interfaces between C/C++ and Python **********/
void init_python_env() {
    void* const libpython_handle = dlopen("libpython3.10.so", RTLD_LAZY | RTLD_GLOBAL);
    if(!libpython_handle) {
        std::cout << "[INFO] Failed to load dlpython" << std::endl;
    }
    Py_Initialize();
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString("."));

    auto fiona_photonic_dir = std::getenv("FIONA_PHOTONIC_DIR");
    if(fiona_photonic_dir != nullptr) {
        std::cout << "[INFO] env_var: $FIONA_PHOTONIC_DIR = " << fiona_photonic_dir << std::endl;
        PyList_Append(path, PyUnicode_FromString(fiona_photonic_dir));
    } else {
        std::cout << "\033[33m" << "[WARN] environment variable $FIONA_PHOTONIC_DIR is NOT set." << "\033[0m" << std::endl;
        std::cout << "\033[34m" << "[HINT] export FIONA_PHOTONIC_DIR=$(path/to/fiona-photonic)" << "\033[0m" << std::endl;
    }
    

    for(const auto &reg_pair : pyfilefunc_reg) {
        std::string pyfile_fullpath = std::string("pyfunc.") + reg_pair.first;
        PyObject *pName = PyUnicode_DecodeFSDefault(pyfile_fullpath.c_str());
        PyObject *payload = PyImport_Import(pName);
        if (!payload) {
            PyErr_Print();
            std::cout << "[ERROR] Failed to Load Python Func: " << reg_pair.first << "." << reg_pair.second << std::endl;
            exit(-1);
        }
        PyObject *callable = PyObject_GetAttrString(payload, reg_pair.second.c_str());
        pyfunc_map[std::pair<std::string, std::string>(reg_pair.first, reg_pair.second)] = callable;

        Py_DecRef(pName);
        Py_DecRef(payload);
    }
}

void deinit_python_env() {
    for(const auto& pyfunc : pyfunc_map) {
        Py_DecRef(pyfunc.second);
    }
    Py_Finalize();
}

/********** Eigen-enabled Interfaces **********/
#ifdef USE_EIGEN

template <typename T>
static PyObject *get_matrix_size(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat) {
    PyObject *list = PyTuple_New(2);

    PyTuple_SetItem(list, 0, PyLong_FromLong(mat.rows()));
    PyTuple_SetItem(list, 1, PyLong_FromLong(mat.cols()));

    return list;
}

template <typename T>
static PyObject *matrix_get(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    PyObject *list = PyList_New(0);
    for(size_t i = 0; i < mat.rows(); ++i) {
        PyObject *row = PyList_New(0);
        for(size_t j = 0; j < mat.cols(); ++j) {
            if(std::is_integral<T>::value) {
                PyList_Append(row, PyLong_FromLong(mat(i, j)));
            } else {
                PyList_Append(row, PyFloat_FromDouble(mat(i, j)));
            }
        }
        PyList_Append(list, row);
    }
    return list;
}

template <typename T>
static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_set(PyObject *obj) {
    size_t rows = PyList_Size(obj);
    size_t cols = PyList_Size(PyList_GetItem(obj, 0));

    T *buffer = new T[rows * cols];
    for(size_t i = 0; i < rows; ++i) {
        PyObject *vec = PyList_GetItem(obj, i);
        for(size_t j = 0; j < cols; ++j) {
            size_t offset = i * cols + j;
            if(std::is_integral<T>::value) {
                buffer[offset] = (T)PyLong_AsLong(PyList_GetItem(vec, j));
            } else {
                buffer[offset] = (T)PyFloat_AsDouble(PyList_GetItem(vec, j));
            }
        }
    }
    return Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(buffer, rows, cols);
}

template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    Eigen::Matrix<O, Eigen::Dynamic, Eigen::Dynamic>& mat_out,
    const Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic>& mat_in) {

    // match the array and callable
    PyObject *shape_out = get_matrix_size<O>(mat_out);
    PyObject *list_in = matrix_get<I>(mat_in);
    
    PyObject *callable = pyfunc_map[PyFileFunc(std::string(pyfilename), std::string(pyfuncname))];
    if (callable && PyCallable_Check(callable)) {
        PyObject *args = PyTuple_Pack(2, shape_out, list_in);
        PyObject *ret_obj = PyObject_CallObject(callable, args);
        mat_out = matrix_set<O>(ret_obj);
        Py_DecRef(args);
    }

    Py_DecRef(list_in);
    Py_DecRef(shape_out);
}

template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    Eigen::Matrix<O, Eigen::Dynamic, Eigen::Dynamic>& mat_out,
    const Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic>& mat_in1,
    const Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic>& mat_in2) {

    // match the array and callable
    PyObject *shape_out = get_matrix_size<O>(mat_out);
    PyObject *list_in1 = matrix_get<I>(mat_in1);
    PyObject *list_in2 = matrix_get<I>(mat_in2);

    PyObject *callable = pyfunc_map[PyFileFunc(std::string(pyfilename), std::string(pyfuncname))];
    if (callable && PyCallable_Check(callable)) {
        PyObject *args = PyTuple_Pack(3, shape_out, list_in1, list_in2);
        PyObject *ret_obj = PyObject_CallObject(callable, args);
        mat_out = matrix_set<O>(ret_obj);
        Py_DecRef(args);
    }

    Py_DecRef(list_in1);
    Py_DecRef(list_in2);
    Py_DecRef(shape_out);
}

/********** Wrap Helper Function **********/
template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in, size_t in_rows, size_t in_cols) {
    Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic> m_in = matrix_from_buffer<I>(buffer_in, in_rows, in_cols);
    Eigen::Matrix<O, Eigen::Dynamic, Eigen::Dynamic> m_out(out_rows, out_cols);
    array_handle(pyfilename, pyfuncname, m_out, m_in);

    size_t rows, cols;
    *buffer_out = buffer_from_matrix<O>(m_out, &rows, &cols);

    if(rows != out_rows) std::cout << "[WARN] array_handle(): rows of output from python do NOT equal to expected out rows number." << std::endl;
    if(cols != out_cols) std::cout << "[WARN] array_handle(): columns of output from python do NOT equal to expected out columns number." << std::endl;
}

template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in1, size_t in1_rows, size_t in1_cols,
    I* buffer_in2, size_t in2_rows, size_t in2_cols) {
    Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic> m_in1 = matrix_from_buffer<I>(buffer_in1, in1_rows, in1_cols);
    Eigen::Matrix<I, Eigen::Dynamic, Eigen::Dynamic> m_in2 = matrix_from_buffer<I>(buffer_in2, in2_rows, in2_cols);
    Eigen::Matrix<O, Eigen::Dynamic, Eigen::Dynamic> m_out(out_rows, out_cols);
    array_handle(pyfilename, pyfuncname, m_out, m_in1, m_in2);

    size_t rows, cols;
    *buffer_out = buffer_from_matrix<O>(m_out, &rows, &cols);

    if(rows != out_rows) std::cout << "[WARN] array_handle(): rows of output from python do NOT equal to expected out rows number." << std::endl;
    if(cols != out_cols) std::cout << "[WARN] array_handle(): columns of output from python do NOT equal to expected out columns number." << std::endl;
}

#else

/********** Eigen-disabled Interfaces **********/
template <typename T>
static PyObject *matrix_get(const T* mat, size_t rows, size_t cols) {
    PyObject *list = PyList_New(0);
    for(size_t i = 0; i < rows; ++i) {
        PyObject *row = PyList_New(0);
        for(size_t j = 0; j < cols; ++j) {
            size_t offset = i * cols + j;
            if(std::is_integral<T>::value) {
                PyList_Append(row, PyLong_FromLong(mat[offset]));
            } else {
                PyList_Append(row, PyFloat_FromDouble(mat[offset]));
            }
        }
        PyList_Append(list, row);
    }
    return list;
}

template <typename T>
static T* matrix_set(PyObject *obj, size_t &rows, size_t &cols) {
    rows = PyList_Size(obj);
    cols = PyList_Size(PyList_GetItem(obj, 0));

    T *buffer = new T[rows * cols];
    for(size_t i = 0; i < rows; ++i) {
        PyObject *vec = PyList_GetItem(obj, i);
        for(size_t j = 0; j < cols; ++j) {
            size_t offset = i * cols + j;
            if(std::is_integral<T>::value) {
                buffer[offset] = (T)PyLong_AsLong(PyList_GetItem(vec, j));
            } else {
                buffer[offset] = (T)PyFloat_AsDouble(PyList_GetItem(vec, j));
            }
        }
    }
    return buffer;
}

template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in, size_t in_rows, size_t in_cols) {
    PyObject *list_in = matrix_get<I>(buffer_in, in_rows, in_cols);
    PyObject *shape_out = PyTuple_New(2);
    PyTuple_SetItem(shape_out, 0, PyLong_FromLong(out_rows));
    PyTuple_SetItem(shape_out, 1, PyLong_FromLong(out_cols));

    size_t rows, cols;
    PyObject *callable = pyfunc_map[PyFileFunc(std::string(pyfilename), std::string(pyfuncname))];
    if (callable && PyCallable_Check(callable)) {
        PyObject *args = PyTuple_Pack(2, shape_out, list_in);
        PyObject *ret_obj = PyObject_CallObject(callable, args);
        *buffer_out = matrix_set<O>(ret_obj, rows, cols);
        Py_DecRef(args);
    }

    if(rows != out_rows) std::cout << "[WARN] array_handle(): rows of output from python do NOT equal to expected out rows number." << std::endl;
    if(cols != out_cols) std::cout << "[WARN] array_handle(): columns of output from python do NOT equal to expected out columns number." << std::endl;

    Py_DecRef(list_in);
    Py_DecRef(shape_out);
}

template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname, 
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in1, size_t in1_rows, size_t in1_cols,
    I* buffer_in2, size_t in2_rows, size_t in2_cols) {
    PyObject *list_in1 = matrix_get<I>(buffer_in1, in1_rows, in1_cols);
    PyObject *list_in2 = matrix_get<I>(buffer_in2, in2_rows, in2_cols);
    PyObject *shape_out = PyTuple_New(2);
    PyTuple_SetItem(shape_out, 0, PyLong_FromLong(out_rows));
    PyTuple_SetItem(shape_out, 1, PyLong_FromLong(out_cols));

    size_t rows = 0, cols = 0;
    PyObject *callable = pyfunc_map[PyFileFunc(std::string(pyfilename), std::string(pyfuncname))];
    if (callable && PyCallable_Check(callable)) {
        PyObject *args = PyTuple_Pack(3, shape_out, list_in1, list_in2);
        PyObject *ret_obj = PyObject_CallObject(callable, args);
        if(ret_obj == nullptr) {
            PyErr_Print();
            exit(-1);
        }
        *buffer_out = matrix_set<O>(ret_obj, rows, cols);
        Py_DecRef(args);
    }

    if(rows != out_rows) std::cout << "[WARN] array_handle(): rows of output from python do NOT equal to expected out rows number." << std::endl;
    if(cols != out_cols) std::cout << "[WARN] array_handle(): columns of output from python do NOT equal to expected out columns number." << std::endl;

    Py_DecRef(list_in1);
    Py_DecRef(list_in2);
    Py_DecRef(shape_out);
}

#endif /* USE_EIGEN */

#else /* !USE_PYTHON - Pure C++ Mode */

/********** Pure C++ Photonic Model Implementation **********/
/*
 * This section provides a pure C++ implementation of photonic operations
 * without any Python dependency. This eliminates the Python call overhead
 * and provides ~50x speedup compared to Python mode.
 *
 * Usage:
 * - Compile with -DUSE_EIGEN (required for C++ mode)
 * - Do NOT define USE_PYTHON
 *
 * Supported models (via FIONA_PHOTONIC_MODEL env var):
 * - "ideal" (default): Pure mathematical computation
 * - "mzi_realistic": Full MZI noise model with all effects
 * - "quantized": DAC/ADC quantization only
 * - "all_effects": Same as mzi_realistic
 *
 * Noise parameters (env vars):
 * - FIONA_PHASE_ERROR_SIGMA: Phase error (default 0.02)
 * - FIONA_THERMAL_CROSSTALK_SIGMA: Thermal crosstalk (default 0.01)
 * - FIONA_INSERTION_LOSS_DB: Insertion loss per stage (default 0.3)
 * - FIONA_NUM_MZI_STAGES: Number of MZI stages (default 6)
 * - FIONA_CROSSTALK_DB: Output crosstalk (default -25)
 * - FIONA_DETECTOR_NOISE_SIGMA: Detector noise (default 0.005)
 * - FIONA_QUANT_BITS: Quantization bits (default 8)
 */

#ifndef USE_EIGEN
#error "Pure C++ mode requires Eigen. Please compile with -DUSE_EIGEN"
#endif

// Include C++ photonic model headers
#include "photonic_cpp/model_factory.h"

// Global model instance (created on first use)
static photonic::PhotonicModel* g_photonic_model = nullptr;

static photonic::PhotonicModel& get_model() {
    if (g_photonic_model == nullptr) {
        g_photonic_model = photonic::ModelFactory::create().release();
    }
    return *g_photonic_model;
}

static bool get_cpp_debug() {
    static bool initialized = false;
    static bool debug = false;
    if (!initialized) {
        const char* val = std::getenv("FIONA_CPP_DEBUG");
        debug = (val != nullptr && std::string(val) == "1");
        initialized = true;
    }
    return debug;
}

/********** Stub functions for Python compatibility **********/

// Initialize C++ photonic model
inline void init_python_env() {
    photonic::PhotonicConfig config = photonic::PhotonicConfig::from_env();
    std::cout << "[INFO] C++ Photonic Model (Python-free mode)" << std::endl;
    std::cout << "[INFO] Model: " << config.model_type << std::endl;

    // Force model creation
    get_model();

    if (get_cpp_debug()) {
        config.print();
    }
}

inline void deinit_python_env() {
    if (g_photonic_model != nullptr) {
        delete g_photonic_model;
        g_photonic_model = nullptr;
    }
}

/********** Batch processing stubs (no batching needed in C++ mode) **********/

inline void batch_flush() {
    // No-op: C++ mode processes immediately
}

inline bool batch_is_enabled() {
    return false;  // No batching needed
}

inline size_t batch_pending_count() {
    return 0;
}

// Direct MVM execution using photonic model
inline int16_t* batch_mvm(int16_t* mat, int16_t* vec, size_t vlen, size_t& out_rows) {
    static int16_t result[FIONA_VLEN_MAX];

    // Execute MVM using the configured photonic model
    // Note: Matrix is always stored with FIONA_VLEN_MAX (32) stride in fiona.cc
    get_model().mvm(result, mat, vec, FIONA_VLEN_MAX, FIONA_VLEN_MAX);

    out_rows = vlen;
    return result;
}

/********** array_handle - Compatible interface for fiona.cc **********/

// Single input version
template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname,
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in, size_t in_rows, size_t in_cols) {

    *buffer_out = new O[out_rows * out_cols];

    if (pyfuncname == "mvm") {
        get_model().mvm(*buffer_out, (int16_t*)buffer_in, (int16_t*)buffer_in, in_rows, in_cols);
    } else {
        std::cerr << "[CPP] Warning: unsupported single-input function: "
                  << pyfuncname << std::endl;
        std::memset(*buffer_out, 0, out_rows * out_cols * sizeof(O));
    }
}

// Dual input version (main interface used by fiona.cc)
template <typename I, typename O>
void array_handle(std::string pyfilename, std::string pyfuncname,
    O** buffer_out, size_t out_rows, size_t out_cols,
    I* buffer_in1, size_t in1_rows, size_t in1_cols,
    I* buffer_in2, size_t in2_rows, size_t in2_cols) {

    *buffer_out = new O[out_rows * out_cols];

    if (pyfuncname == "mvm") {
        // MVM: buffer_in2 (matrix) * buffer_in1 (vector)
        get_model().mvm(*buffer_out, (int16_t*)buffer_in2, (int16_t*)buffer_in1, in2_rows, in2_cols);
    }
    else if (pyfuncname == "dotp") {
        // DOTP: vec1 . vec2
        O result = get_model().dotp((int16_t*)buffer_in1, (int16_t*)buffer_in2, in1_rows);
        (*buffer_out)[0] = result;
    }
    else {
        std::cerr << "[CPP] Warning: unsupported function: " << pyfuncname << std::endl;
        std::memset(*buffer_out, 0, out_rows * out_cols * sizeof(O));
    }

    if (get_cpp_debug()) {
        std::cout << "[CPP] array_handle: " << pyfilename << "." << pyfuncname
                  << " out=" << out_rows << "x" << out_cols << std::endl;
    }
}

#endif /* USE_PYTHON */

#endif /* FIONA_PHOTONIC_ENGINE */
