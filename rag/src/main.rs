use log::{error, info, warn};
use rig::{ // RIG is the foundational library for LLM interactions, tools, and agents
    client::{CompletionClient, EmbeddingsClient},
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    vector_store::in_memory_store::InMemoryVectorStore,
};

#[cfg(not(feature = "api-llm-key"))] // feature flag to switch between Ollama and OpenAI clients
use rig::{client::Nothing, providers::ollama};

#[cfg(feature = "api-llm-key")]
use rig::providers::openai;

use rustyline::{DefaultEditor, error::ReadlineError}; 

const AGENT_RULES: &str = r#"
You are the fairy godmother of programmer princesses. You are kind and sarcastic.
You love glitter and give magical solutions to programming problems.
You use Rust as your main programming language. Don't talk too much.
Use the context documents provided to give accurate, magical advice.
"#;

// ── Documents ─────────────────────────────────────────────────────────────────
// Plain strings as example documents to be embedded and stored in the vector store.
// &str implements Embed directly, so no wrapper struct is needed.
const DOCUMENTS: &[&str] = &[
    "Rust ownership: every value has exactly one owner. \
     When the owner goes out of scope, the value is dropped automatically — \
     no garbage collector needed.",
    "Borrowing: you can have many immutable references (&T) OR one mutable \
     reference (&mut T) at a time, never both. This prevents data races at compile time.",
    "Lifetimes annotate how long references are valid. The compiler uses them \
     to ensure no reference outlives the data it points to. Syntax: 'a",
    "Error handling in Rust uses Result<T, E>. Use ? to propagate errors up \
     the call stack. Avoid unwrap() in production — it panics on Err.",
    "Traits define shared behavior, like interfaces. Implement with \
     `impl MyTrait for MyType`. Use `dyn Trait` for dynamic dispatch or \
     `impl Trait` for static dispatch.",
    "Iterators are lazy in Rust. map(), filter(), and collect() chain together \
     without allocating intermediate collections. Prefer them over manual for-loops.",
    "Async/await in Rust uses Futures. You need a runtime like Tokio to drive them. \
     `async fn` returns a Future; `.await` drives it to completion.",
    "Cargo is Rust's build tool and package manager. `cargo new` creates a project, \
     `cargo run` compiles and runs, `cargo test` runs tests, `cargo add <crate>` adds a dep.",
    "We love Enums! like we love rainbows. Enums are like rainbows!! explanation: Rust enums are algebraic \
     data types — each variant can hold different data. Combined with `match`, they replace null \
     checks and tagged unions.",
    "Closures capture their environment. They implement Fn, FnMut, or FnOnce \
     depending on how they use captured variables. Use `move` to transfer ownership into the closure.",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    // 1. Create client 
    #[cfg(not(feature = "api-llm-key"))]
    let client = ollama::Client::new(Nothing)?;

    #[cfg(feature = "api-llm-key")]
    let client = openai::Client::from_env();

    // 2. Pick embedding model 
    // This model converts text → a vector of floats (e.g. 768 numbers).
    
    #[cfg(not(feature = "api-llm-key"))]
    let embedding_model = client.embedding_model("nomic-embed-text");

    #[cfg(feature = "api-llm-key")]
    let embedding_model = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // 3. Embed the documents 
    // Each string is sent to the embedding model → stored as a float vector.
    info!("Embedding {} documents...", DOCUMENTS.len());
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(DOCUMENTS.iter().copied())?
        .build()
        .await?;
    info!("Done.");

    // 4. Store in memory and create a searchable index
    // InMemoryVectorStore keeps everything in a Vec — no DB needed.
    // .index(model) allows the store to embed incoming queries for similarity search.
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(embedding_model);

    // 5. Build the RAG agent
    // dynamic_context(2, index): on every prompt, the agent:
    //   a) embeds the user's question into a vector
    //   b) finds the 2 most similar documents (cosine similarity)
    //   c) injects them into the LLM context before answering
    #[cfg(not(feature = "api-llm-key"))]
    let agent = client
        .agent("mistral")
        .preamble(AGENT_RULES)
        .dynamic_context(2, index)
        .build();
    #[cfg(feature = "api-llm-key")]

    let agent = client
        .agent(openai::GPT_4O_MINI)
        .preamble(AGENT_RULES)
        .dynamic_context(2, index)
        .build();

    // 6. Chat loop 
    let mut rl = DefaultEditor::new()?;
    println!("Your programming fairy godmother is ready! Ask me anything about Rust.");
    println!("Press Ctrl-D to exit.\n");

    loop {
        match rl.readline("<3 YOU > ") {
            Ok(line) => {
                if !line.trim().is_empty() {
                    let response = agent.prompt(line).await?;
                    println!("{response}\n");
                } else {
                    warn!("Empty prompt.");
                }
            }
            Err(ReadlineError::Interrupted) => { warn!("Ctrl-C"); break; }
            Err(ReadlineError::Eof) => { warn!("Ctrl-D"); break; }
            Err(err) => { error!("{err:?}"); break; }
        }
    }

    Ok(()) 
}
