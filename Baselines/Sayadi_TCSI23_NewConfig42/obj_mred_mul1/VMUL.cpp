// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "VMUL__pch.h"

//============================================================
// Constructors

VMUL::VMUL(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new VMUL__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , a{vlSymsp->TOP.a}
    , b{vlSymsp->TOP.b}
    , out{vlSymsp->TOP.out}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

VMUL::VMUL(const char* _vcname__)
    : VMUL(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

VMUL::~VMUL() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void VMUL___024root___eval_debug_assertions(VMUL___024root* vlSelf);
#endif  // VL_DEBUG
void VMUL___024root___eval_static(VMUL___024root* vlSelf);
void VMUL___024root___eval_initial(VMUL___024root* vlSelf);
void VMUL___024root___eval_settle(VMUL___024root* vlSelf);
void VMUL___024root___eval(VMUL___024root* vlSelf);

void VMUL::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VMUL::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    VMUL___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        VMUL___024root___eval_static(&(vlSymsp->TOP));
        VMUL___024root___eval_initial(&(vlSymsp->TOP));
        VMUL___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    VMUL___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool VMUL::eventsPending() { return false; }

uint64_t VMUL::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* VMUL::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void VMUL___024root___eval_final(VMUL___024root* vlSelf);

VL_ATTR_COLD void VMUL::final() {
    VMUL___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* VMUL::hierName() const { return vlSymsp->name(); }
const char* VMUL::modelName() const { return "VMUL"; }
unsigned VMUL::threads() const { return 1; }
void VMUL::prepareClone() const { contextp()->prepareClone(); }
void VMUL::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void VMUL::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'VMUL::trace()' called on model that was Verilated without --trace option");
}
