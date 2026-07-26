// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VMUL__SYMS_H_
#define VERILATED_VMUL__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VMUL.h"

// INCLUDE MODULE CLASSES
#include "VMUL___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)VMUL__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VMUL* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    VMUL___024root                 TOP;

    // CONSTRUCTORS
    VMUL__Syms(VerilatedContext* contextp, const char* namep, VMUL* modelp);
    ~VMUL__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
