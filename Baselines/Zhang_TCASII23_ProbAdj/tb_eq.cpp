#include "VMUL.h"
#include <verilated.h>
#include <cstdio>
int main(int argc, char** argv){
  Verilated::commandArgs(argc, argv);
  VMUL* d = new VMUL;
  FILE* f = fopen(argv[1], "r");
  long a,b,exp,mism=0,cnt=0;
  while(fscanf(f,"%ld %ld %ld",&a,&b,&exp)==3){
    d->a=a; d->b=b; d->clk=0; d->eval();
    long got=d->out;
    if(got!=exp){ if(mism<6) printf("MISMATCH a=%ld b=%ld exp=%ld got=%ld\n",a,b,exp,got); mism++; }
    cnt++;
  }
  printf("checked=%ld mismatches=%ld\n",cnt,mism);
  delete d; return mism?1:0;
}
