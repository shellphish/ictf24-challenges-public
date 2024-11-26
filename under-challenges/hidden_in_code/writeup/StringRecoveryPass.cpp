#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <queue>
#include <random>
#include <string>
#include <set>

using namespace llvm;

namespace {
struct StringRecoveryPass : public FunctionPass {
  static char ID;
  StringRecoveryPass() : FunctionPass(ID) {}

  std::string recoveredString;

  bool runOnFunction(Function &F) override {
    if (F.getName() == "main") 
      startRecover(F);
    return false;
  }

  void startRecover(Function &F) {
    std::queue<Function*> functionQueue;
    std::set<Function*> visitedFunctions;

    functionQueue.push(&F);
    visitedFunctions.insert(&F);

    while (!functionQueue.empty()) {
      Function* currentFunction = functionQueue.front();
      functionQueue.pop();

      for (auto &BB : *currentFunction) {
        recoveredString += recoverMetadata(BB);
      }

      for (auto &BB : *currentFunction) {
        for (auto &I : BB) {
          if (auto *callInst = dyn_cast<CallInst>(&I)) {
            Function *calledFunction = callInst->getCalledFunction();
            if (calledFunction && visitedFunctions.find(calledFunction) == visitedFunctions.end()) {
              functionQueue.push(calledFunction);
              visitedFunctions.insert(calledFunction);
            }
          }
        }
      }
    }

    // Output the recovered string
    errs() << "Recovered string: " << recoveredString << "\n";
  }
  

  std::string recoverMetadata(BasicBlock &BB) {
    if (MDNode *mdNode = BB.getFirstNonPHI()->getMetadata("secret")) {
        if (MDString *mdString = dyn_cast<MDString>(mdNode->getOperand(0))) {
            return mdString->getString().str();
        }
    }
  }
};
}

char StringRecoveryPass::ID = 0;
static RegisterPass<StringRecoveryPass> X("string-recovery", "String Recovery Pass");

static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
  [](const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
    PM.add(new StringRecoveryPass());
  });