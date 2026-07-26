// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VMUL.h for the primary calling header

#ifndef VERILATED_VMUL___024ROOT_H_
#define VERILATED_VMUL___024ROOT_H_  // guard

#include "verilated.h"


class VMUL__Syms;

class alignas(VL_CACHE_LINE_BYTES) VMUL___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __VactContinue;
    VL_IN16(a,15,0);
    VL_IN16(b,15,0);
    VL_OUT(out,30,0);
    IData/*31:0*/ __VactIterCount;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<0> __VactTriggered;
    VlTriggerVec<0> __VnbaTriggered;

    // INTERNAL VARIABLES
    VMUL__Syms* const vlSymsp;

    // CONSTRUCTORS
    VMUL___024root(VMUL__Syms* symsp, const char* v__name);
    ~VMUL___024root();
    VL_UNCOPYABLE(VMUL___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
