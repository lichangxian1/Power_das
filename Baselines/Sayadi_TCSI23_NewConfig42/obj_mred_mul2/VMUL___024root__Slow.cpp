// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMUL.h for the primary calling header

#include "VMUL__pch.h"
#include "VMUL__Syms.h"
#include "VMUL___024root.h"

void VMUL___024root___ctor_var_reset(VMUL___024root* vlSelf);

VMUL___024root::VMUL___024root(VMUL__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VMUL___024root___ctor_var_reset(this);
}

void VMUL___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

VMUL___024root::~VMUL___024root() {
}
