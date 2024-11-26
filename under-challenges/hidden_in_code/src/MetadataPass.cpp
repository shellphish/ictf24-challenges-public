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
struct MetadataPass : public FunctionPass {
  static char ID;
  MetadataPass() : FunctionPass(ID) {}

  const std::string secretString = "qwertyuiopasdfghjklzxcvbnm0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ iCTF{fake_flag}";
  size_t currentCharIndex = 0;
  std::set<BasicBlock*> processedBlocks;

  bool runOnFunction(Function &F) override {
    if (F.getName() == "main") {
      processMainFunction(F);
    } else {
      processOtherFunction(F);
    }
    return true;
  }

  void processMainFunction(Function &F) {
    std::queue<Function*> functionQueue;
    std::set<Function*> visitedFunctions;

    functionQueue.push(&F);
    visitedFunctions.insert(&F);

    while (!functionQueue.empty() && currentCharIndex < secretString.length()) {
      Function* currentFunction = functionQueue.front();
      functionQueue.pop();

      for (auto &BB : *currentFunction) {
        if (currentCharIndex < secretString.length() && processedBlocks.find(&BB) == processedBlocks.end()) {
          insertMetadata(BB, std::string(1, secretString[currentCharIndex++]));
          processedBlocks.insert(&BB);
        } else {
          break;
        }
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
  }

  void processOtherFunction(Function &F) {
    for (auto &BB : F) {
      if (processedBlocks.find(&BB) == processedBlocks.end()) {
        insertMetadata(BB, generateRandomString(8));
        processedBlocks.insert(&BB);
      }
    }
  }

  void insertMetadata(BasicBlock &BB, const std::string &value) {
    LLVMContext &context = BB.getContext();
    MDNode *mdNode = MDNode::get(context, MDString::get(context, value));
    BB.getInstList().front().setMetadata("secret", mdNode);
  }

  std::string generateRandomString(size_t length) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distribution(0, charset.size() - 1);

    std::string result;
    for (size_t i = 0; i < length; ++i) {
      result += charset[distribution(generator)];
    }
    return result;
  }
};
}

char MetadataPass::ID = 0;

static RegisterPass<MetadataPass> X("metadata-pass", "Metadata Insertion Pass");

static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
  [](const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
    PM.add(new MetadataPass());
  });
