use llm::{ // LLM interaction library
    LLMProvider, // Trait for LLM interactions
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
}; 
use log::{error, info};

#[tokio::main] 
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    #[cfg(not(feature = "api-llm-key"))] // If the "api-llm-key" feature is not enabled, use the Ollama client
    info!("Using =Ollama=");

    // Initialize and configure the LLM client
    let llm = build_llm();

    let mut rl = rustyline::DefaultEditor::new()?;

    let mut messages: Vec<ChatMessage> = vec![]; // vector to hold the conversation history

    loop {
        let input = match rl.readline("<3 YOU > ") {
            Ok(line) => line, 
            Err(_) => break, // Ctrl+C or Ctrl+D exits
        };

        if input.trim().is_empty() { continue; }

        // Append user message to history
        messages.push(ChatMessage::user().content(&input).build());

        // Send full history to LLM
        match llm.chat(&messages).await { 
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("Bot: {text}");
                    // Append assistant response to history (context!)
                    messages.push(ChatMessage::assistant().content(text).build());
                }
            }
            Err(e) => error!("Chat error: {e}"),
        }
    }

    Ok(())
}

#[cfg(feature = "api-llm-key")]
fn build_llm() -> Box<dyn LLMProvider> {
    let api_key = std::env::var("API_KEY").unwrap_or("sk-TESTKEY".into());

    LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider (example)
        .api_key(api_key) // Set the API key
        .model("gpt-4.1-nano") // Use GPT-4.1 Nano model
        .max_tokens(512) // Limit response length
        .temperature(0.7)
        .normalize_response(true) 
        .system("You are the fairy godmother of programmer princesses. You are kind and sarcastic. You love glitter and give magical solutions to programming problems. You use Rust as your main programming language.")
        .build()
        .expect("Failed to build LLM")
}

#[cfg(not(feature = "api-llm-key"))]
fn build_llm() -> Box<dyn LLMProvider> {
    LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM provider
        .model("mistral") // Use Mistral model (example)
        .max_tokens(512)
        .temperature(0.7)
        .normalize_response(true)
        .system("You are the fairy godmother of programmer princesses. You are kind and sarcastic. You love glitter and give magical solutions to programming problems. You use Rust as your main programming language. Don't talk too much")
        .build()
        .expect("Failed to build LLM")
}