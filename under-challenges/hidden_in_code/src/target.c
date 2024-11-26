#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Define the maximum length for user input
#define MAX_INPUT_LENGTH 256

// Function prototypes for all 20 questions
void askQuestion1();
void askQuestion2();
void askQuestion3();
void askQuestion4();
void askQuestion5();
void askQuestion6();
void askQuestion7();
void askQuestion8();
void askQuestion9();
void askQuestion10();
void askQuestion11();
void askQuestion12();
void askQuestion13();
void askQuestion14();
void askQuestion15();
void askQuestion16();
void askQuestion17();
void askQuestion18();
void askQuestion19();
void askQuestion20();

// Function to remove the newline character from the input
void removeNewline(char *str) {
    size_t len = strlen(str);
    if(len > 0 && str[len-1] == '\n') {
        str[len-1] = '\0';
    }
}

int main() {
    // Start with the first question
    askQuestion1();
    return 0;
}

// Function to ask the first question
void askQuestion1() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 1: What does AI stand for?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "Artificial Intelligence") == 0) {
            printf("Correct!\n\n");
            askQuestion2();
        } else {
            printf("Incorrect. The correct answer is 'Artificial Intelligence'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the second question
void askQuestion2() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 2: Who is known as the father of Artificial Intelligence?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "John McCarthy") == 0) {
            printf("Correct!\n\n");
            askQuestion3();
        } else {
            printf("Incorrect. The correct answer is 'John McCarthy'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the third question
void askQuestion3() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 3: What is the Turing Test used for?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "intelligence") != NULL || strstr(answer, "conversation") != NULL) {
            printf("Correct! The Turing Test is used to determine if a machine exhibits human-like intelligence.\n\n");
            askQuestion4();
        } else {
            printf("Incorrect. The Turing Test is used to determine if a machine exhibits human-like intelligence.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the fourth question
void askQuestion4() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 4: In which year was the term 'Artificial Intelligence' coined?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "1956") == 0) {
            printf("Correct!\n\n");
            askQuestion5();
        } else {
            printf("Incorrect. The correct answer is '1956'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the fifth question
void askQuestion5() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 5: Name an AI programming language developed in the 1950s.\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "LISP") == 0 || strcmp(answer, "Lisp") == 0) {
            printf("Correct!\n\n");
            askQuestion6();
        } else {
            printf("Incorrect. The correct answer is 'LISP'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the sixth question (LLM-related)
void askQuestion6() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 6: What is the purpose of attention mechanisms in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "focus") != NULL && strstr(answer, "relevant parts") != NULL) {
            printf("Correct! Attention mechanisms allow the model to focus on relevant parts of the input when generating each part of the output.\n\n");
            askQuestion7();
        } else {
            printf("Incorrect. Attention mechanisms allow the model to focus on relevant parts of the input when generating each part of the output.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the seventh question (LLM-related)
void askQuestion7() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 7: What is the primary architectural difference between GPT and BERT models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if ((strstr(answer, "unidirectional") != NULL && strstr(answer, "left-to-right") != NULL) ||
            (strstr(answer, "bidirectional") != NULL && strstr(answer, "understanding") != NULL)) {
            printf("Correct! GPT is a unidirectional (left-to-right) model primarily for text generation, whereas BERT is bidirectional, primarily for understanding tasks.\n\n");
            askQuestion8();
        } else {
            printf("Incorrect. GPT is a unidirectional (left-to-right) model primarily for text generation, whereas BERT is bidirectional, primarily for understanding tasks.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the eighth question
void askQuestion8() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 8: Name a famous AI developed by IBM.\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "Watson") == 0) {
            printf("Correct!\n\n");
            askQuestion9();
        } else {
            printf("Incorrect. The correct answer is 'Watson'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the ninth question (LLM-related)
void askQuestion9() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 9: What is the role of a tokenizer in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "convert") != NULL && strstr(answer, "tokens") != NULL) {
            printf("Correct! A tokenizer converts text into tokens that the model can process.\n\n");
            askQuestion10();
        } else {
            printf("Incorrect. A tokenizer converts text into tokens that the model can process.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the tenth question
void askQuestion10() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 10: Who wrote the book 'Artificial Intelligence: A Modern Approach'?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strcmp(answer, "Stuart Russell and Peter Norvig") == 0 ||
            strcmp(answer, "Stuart Russell") == 0 ||
            strcmp(answer, "Peter Norvig") == 0) {
            printf("Correct!\n\n");
            askQuestion11();
        } else {
            printf("Incorrect. The correct answer is 'Stuart Russell and Peter Norvig'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the eleventh question (LLM-related)
void askQuestion11() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 11: How does fine-tuning differ from pre-training in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "pre-training") != NULL && strstr(answer, "fine-tuning") != NULL) {
            printf("Correct! Pre-training involves training on a large corpus for general language understanding, while fine-tuning adapts the model to specific tasks with additional training.\n\n");
            askQuestion12();
        } else {
            printf("Incorrect. Pre-training involves training on a large corpus for general language understanding, while fine-tuning adapts the model to specific tasks with additional training.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the twelfth question
void askQuestion12() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 12: Name a virtual assistant powered by AI.\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Accept multiple correct answers
        if (strcmp(answer, "Siri") == 0 || strcmp(answer, "Alexa") == 0 ||
            strcmp(answer, "Google Assistant") == 0 || strcmp(answer, "Cortana") == 0) {
            printf("Correct!\n\n");
            askQuestion13();
        } else {
            printf("Incorrect. Acceptable answers include 'Siri', 'Alexa', 'Google Assistant', or 'Cortana'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the thirteenth question (LLM-related)
void askQuestion13() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 13: What is few-shot learning in the context of large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "few examples") != NULL || strstr(answer, "prompt") != NULL) {
            printf("Correct! Few-shot learning refers to the ability of a model to perform tasks with only a few examples provided in the prompt.\n\n");
            askQuestion14();
        } else {
            printf("Incorrect. Few-shot learning refers to the ability of a model to perform tasks with only a few examples provided in the prompt.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the fourteenth question (LLM-related)
void askQuestion14() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 14: What is the significance of the transformer architecture in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "parallel processing") != NULL || strstr(answer, "self-attention") != NULL) {
            printf("Correct! The transformer architecture enables parallel processing and efficient handling of long-range dependencies through self-attention mechanisms.\n\n");
            askQuestion15();
        } else {
            printf("Incorrect. The transformer architecture enables parallel processing and efficient handling of long-range dependencies through self-attention mechanisms.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the fifteenth question (LLM-related)
void askQuestion15() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 15: What is prompt engineering in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "designing prompts") != NULL || strstr(answer, "guide outputs") != NULL) {
            printf("Correct! Prompt engineering involves designing input prompts to guide the model to generate desired outputs.\n\n");
            askQuestion16();
        } else {
            printf("Incorrect. Prompt engineering involves designing input prompts to guide the model to generate desired outputs.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the sixteenth question
void askQuestion16() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 16: Name a popular AI framework.\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Accept multiple correct answers
        if (strcmp(answer, "TensorFlow") == 0 || strcmp(answer, "PyTorch") == 0 ||
            strcmp(answer, "Keras") == 0 || strcmp(answer, "Caffe") == 0) {
            printf("Correct!\n\n");
            askQuestion17();
        } else {
            printf("Incorrect. Acceptable answers include 'TensorFlow', 'PyTorch', 'Keras', or 'Caffe'.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the seventeenth question
void askQuestion17() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 17: What is the Singularity in AI?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        if (strstr(answer, "future") != NULL && strstr(answer, "intelligence") != NULL) {
            printf("Correct! The Singularity refers to a hypothetical future point where AI surpasses human intelligence, leading to exponential technological growth.\n\n");
            askQuestion18();
        } else {
            printf("Incorrect. The Singularity refers to a hypothetical future point where AI surpasses human intelligence, leading to exponential technological growth.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the eighteenth question (LLM-related)
void askQuestion18() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 18: What is the purpose of using beam search in large language models?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "coherent") != NULL || strstr(answer, "multiple outputs") != NULL) {
            printf("Correct! Beam search is used to generate more coherent and likely sequences by exploring multiple possible outputs.\n\n");
            askQuestion19();
        } else {
            printf("Incorrect. Beam search is used to generate more coherent and likely sequences by exploring multiple possible outputs.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the nineteenth question (LLM-related)
void askQuestion19() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 19: What is model scaling in large language models and why is it important?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "increase size") != NULL && strstr(answer, "performance") != NULL) {
            printf("Correct! Model scaling involves increasing the size of the model (parameters, data) to improve performance.\n\n");
            askQuestion20();
        } else {
            printf("Incorrect. Model scaling involves increasing the size of the model (parameters, data) to improve performance.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

// Function to ask the twentieth question (LLM-related)
void askQuestion20() {
    char answer[MAX_INPUT_LENGTH];

    printf("Question 20: What are some common challenges in deploying large language models in production environments?\n");
    printf("Your answer: ");
    if (fgets(answer, sizeof(answer), stdin) != NULL) {
        removeNewline(answer);
        // Checking for key keywords
        if (strstr(answer, "latency") != NULL || strstr(answer, "scalability") != NULL ||
            strstr(answer, "cost") != NULL || strstr(answer, "biases") != NULL) {
            printf("Correct! Common challenges include latency, scalability, cost, and handling biases.\n\n");
            printf("Congratulations! You have answered all 20 questions correctly.\n");
            printf("I will give the flag\n");
            printf("iCTF{7hi5_1s_a_f@ke_flag}\n");
            printf("Just kidding.");
        } else {
            printf("Incorrect. Common challenges include latency, scalability, cost, and handling biases.\n");
        }
    } else {
        printf("Error reading input.\n");
    }
}

