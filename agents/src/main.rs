use std::{ 
    fs::{self, read_to_string},
    path::Path,
};

use anyhow::{Result, bail}; // Error handling library

use log::{debug, error, info, warn}; 

use rig::{ // RIG is the foundational library for LLM interactions, tools, and agents
    client::CompletionClient,
    completion::{Completion, Message},
    message::AssistantContent,
};

#[cfg(not(feature = "api-llm-key"))]
use rig::{client::Nothing, providers::ollama};

#[cfg(feature = "api-llm-key")]
use rig::providers::openai;

use rustyline::{DefaultEditor, error::ReadlineError};

use serde::Deserialize;
use serde_json::{Map, Value, json};

// Action struct represents a tool call requested by the LLM
// The LLM must produce this JSON inside an action block to invoke a tool.
#[derive(Debug, Deserialize)] // Deserialize from JSON
struct Action {
    tool: String, // Name of the tool to call, e.g. "list_files"
    #[serde(default)] // Default to empty object if "args" is missing
    args: Map<String, Value>, // Arguments for the tool, e.g. {"file_name": "secret.txt"}
}

// System prompt
// Tells the LLM which tools exist and exactly how to call them.
const AGENT_RULES: &str = r#"
You are the fairy godmother of programmer princesses. You are kind and sarcastic.
You love glitter and give magical solutions to programming problems.
You use Rust as your main programming language. Don't talk too much.

You have a magical wand with the following spells (tools):
- list_files  : reveals all files hidden in the current directory. Args: none.
- read_file   : opens and reads the enchanted scroll at the given path. Args: {"file_name": "<path>"}
- terminate   : ends the spell and delivers your final magical answer. Args: {"message": "<your answer>"}

To cast a spell, respond ONLY with a fenced ```action block — no other text, no glitter yet:

```action
{"tool": "list_files", "args": {}}
```

After seeing the spell result, keep reasoning and cast another spell, or cast
terminate when you have enough magic to answer the princess.
If you need no spells, cast terminate immediately with your answer.
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    #[cfg(not(feature = "api-llm-key"))]
    info!("Using =Ollama=");

    #[cfg(not(feature = "api-llm-key"))]

    let agent = {
        let client: ollama::Client = ollama::Client::new(Nothing)?;
        client
            .agent("llama3.2") // Mistral does not support tool calling
            .preamble(AGENT_RULES)
            .build()
    };

    #[cfg(feature = "api-llm-key")]
    let agent = openai::Client::from_env()
        .agent(openai::GPT_4O)
        .preamble(AGENT_RULES)
        .build();

    let mut chat_history: Vec<Message> = vec![];

    let mut rl = DefaultEditor::new()?;
    println!("Your fairy godmother is here\n");

    // Outer loop: wait for user input
    loop {
        match rl.readline("<3 YOU > ") {
            Ok(line) => {
                if line.trim().is_empty() {
                    warn!("Empty prompt.");
                    continue;
                }
                // Inner agentic loop: Look for the correct tools, and tool results until termination
                match run_agent(&agent, &line, &mut chat_history).await {
                    Ok(answer) => println!("\n{answer}\n"),
                    Err(e) => error!("Agent error: {e}"),
                }
            }
            Err(ReadlineError::Interrupted) => { warn!("Ctrl-C"); break; }
            Err(ReadlineError::Eof) => { warn!("Ctrl-D"); break; }
            Err(err) => { error!("{err:?}"); break; }
        }
    }

    Ok(())
}

// Agentic loop
// Calls the LLM, executes any tool it requests, feeds the result back, and
// repeats until the LLM calls `terminate` (or produces no action block).
async fn run_agent<M>(
    agent: &rig::agent::Agent<M>,
    user_prompt: &str,
    history: &mut Vec<Message>,
) -> Result<String>
where
    M: rig::completion::CompletionModel,
{
    let mut current_prompt = user_prompt.to_string();

    loop {

        let completion = agent
            .completion(&current_prompt, history.to_owned())
            .await?
            .send()
            .await?;

        let AssistantContent::Text(text_content) = completion.choice.first() else {
            bail!("Non-text response: {:?}", completion.choice);
        }; 
        let llm_text = text_content.text();
        debug!("LLM said: {llm_text}");

        match parse_action(llm_text) {
            Ok(action) => {
                info!("→ tool: {} | args: {:?}", action.tool, action.args);

                // `terminate` = LLM has a final answer
                if action.tool == "terminate" {
                    let answer = action.args
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or(llm_text)
                        .to_string();
                    history.push(Message::user(&current_prompt));
                    history.push(Message::assistant(llm_text));
                    return Ok(answer);
                }

                // Execute the tool
                let result = dispatch_tool(&action);
                info!("← result: {}", serde_json::to_string(&result)?);

                // Commit this exchange to history, then loop with the tool result as next prompt
                history.push(Message::user(&current_prompt));
                history.push(Message::assistant(llm_text));
                current_prompt = format!(
                    "Tool result:\n```json\n{}\n```",
                    serde_json::to_string_pretty(&result)?
                );
            }
            // No action block → plain answer, no tool needed
            Err(_) => {
                history.push(Message::user(&current_prompt));
                history.push(Message::assistant(llm_text));
                return Ok(llm_text.to_string());
            }
        }
    }
}

// Tool dispatcher 
fn dispatch_tool(action: &Action) -> Value {
    match action.tool.as_str() {
        "list_files" => list_files(),
        "read_file" => {
            let filename = action.args
                .get("file_name")
                .and_then(Value::as_str)
                .unwrap_or("");
            read_file(filename)
        }
        "terminate" => terminate(
            action.args
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Done"),
        ),
        unknown => json!({"error": format!("Unknown tool: {unknown}")}),
    }
}

// Tools
fn list_files() -> Value {
    match fs::read_dir(Path::new(".")) {
        Ok(paths) => {
            let files = paths
                .flatten()
                .map(|e| format!("{:?}", e.path()))
                .collect::<Vec<_>>();
            json!({"tool": "list_files", "result": files})
        }
        Err(err) => json!({"tool": "list_files", "error": err.to_string()}),
    }
}

fn read_file<P: AsRef<Path>>(filename: P) -> Value {
    debug!("Reading: {:?}", filename.as_ref());
    match read_to_string(&filename) {
        Ok(content) => json!({
            "tool": "read_file",
            "file_name": filename.as_ref().to_str().unwrap_or_default(),
            "result": content
        }),
        Err(err) => json!({
            "tool": "read_file",
            "file_name": filename.as_ref().to_str().unwrap_or_default(),
            "error": err.to_string()
        }),
    }
}

fn terminate(msg: &str) -> Value {
    debug!("Terminate: {msg}");
    Value::Object(Map::new())
}

// Action parser
fn parse_action(response: &str) -> Result<Action> {
    let Some(block) = extract_action_block(response) else {
        bail!("No action block found");
    };
    debug!("Parsing action: {block}");
    Ok(serde_json::from_str(&block)?)
}


pub fn extract_action_block(markdown: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut in_block = false;

    for line in markdown.lines() {
        if in_block {
            if line.trim().starts_with("```") {
                return Some(lines.join("\n"));
            }
            lines.push(line.to_string());
        } else if line.trim() == "```action" {
            in_block = true;
            lines.clear();
        }
    }
    None
}
