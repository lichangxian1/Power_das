// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMUL.h for the primary calling header

#include "VMUL__pch.h"
#include "VMUL__Syms.h"
#include "VMUL___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void VMUL___024root___dump_triggers__ico(VMUL___024root* vlSelf);
#endif  // VL_DEBUG

void VMUL___024root___eval_triggers__ico(VMUL___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMUL__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMUL___024root___eval_triggers__ico\n"); );
    // Body
    vlSelf->__VicoTriggered.set(0U, (IData)(vlSelf->__VicoFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        VMUL___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VMUL___024root___dump_triggers__act(VMUL___024root* vlSelf);
#endif  // VL_DEBUG

void VMUL___024root___eval_triggers__act(VMUL___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMUL__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMUL___024root___eval_triggers__act\n"); );
    // Body
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        VMUL___024root___dump_triggers__act(vlSelf);
    }
#endif
}
