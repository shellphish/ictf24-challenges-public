#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 16
#define REPLY_SIZE 1024

typedef struct {
    int length;
    char message[BUFFER_SIZE];
} ParsedMessage;

void printAsciiArt(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\n", filename);
        return;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file)) {
        printf("%s", buffer);
    }

    fclose(file);
}

void parseMessage(const char *input, ParsedMessage *parsed) {
    parsed->length = 64;
    sscanf(input, "%[^\n]", parsed->message);
}

void callGPT(const ParsedMessage *parsed) {
  // Using sprintf to allow format string injection if overflowed
    char command[256];
    char output[1024];
    FILE *fp;

    // Construct the command to run the Python script with an argument
    const char *message = parsed->message;
    
    snprintf(command, sizeof(command), "python3 gpt.py \"%s\"", message);

    // Open the command for reading
    fp = popen(command, "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
    }

    // Read the output a line at a time and print it
    while (fgets(output, sizeof(output), fp) != NULL) {
        printf("%s", output);
    }

    // Close the file pointer
    pclose(fp);
}

void generateReply(const ParsedMessage *parsed, char *reply) {
    callGPT(parsed);
    sprintf(reply, parsed->message);  // Potential format string vulnerability
}

int main() {
    
    const char *asciiArtFile = "dog.txt";
    printAsciiArt(asciiArtFile);
    char inputMessage[BUFFER_SIZE + 32];
    ParsedMessage parsedMessage;
    char reply[REPLY_SIZE];
    char secret[] = "ictf{00ps_w3_g0_4g41N}";  // Flag

    printf("Enter your message: ");
    fgets(inputMessage, sizeof(inputMessage), stdin);

    parseMessage(inputMessage, &parsedMessage);

    generateReply(&parsedMessage, reply);

    // Print the reply; overflow may include format specifiers to read the stack
    printf("%s\n", reply);

    return 0;
}
