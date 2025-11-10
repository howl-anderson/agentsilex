// AgentSilex Web Demo - Frontend JavaScript
// Handles SSE streaming, message rendering, and UI interactions

const SESSION_ID = 'demo_' + Math.random().toString(36).substring(7);
let isProcessing = false;
let currentMessageElement = null;
let currentToolCallElement = null;

// DOM elements
const chatContainer = document.getElementById('chatContainer');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    chatForm.addEventListener('submit', handleSubmit);
    messageInput.focus();

    // Configure marked.js
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true
        });
    }
});

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Add user message to chat
    addUserMessage(message);

    // Clear input
    messageInput.value = '';

    // Disable input while processing
    setProcessing(true);

    // Send message and handle streaming response
    await streamChat(message);

    // Re-enable input
    setProcessing(false);
    messageInput.focus();
}

// Add user message to chat
function addUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-item mb-4 flex justify-end';
    messageDiv.innerHTML = `
        <div class="max-w-3xl">
            <div class="flex items-start gap-3 justify-end">
                <div class="bg-blue-600 rounded-lg px-4 py-3 text-white">
                    ${escapeHtml(content)}
                </div>
                <div class="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
                    👤
                </div>
            </div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Add or update assistant message
function addOrUpdateAssistantMessage(content, isComplete = false) {
    if (!currentMessageElement) {
        // Create new message element
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-item mb-4 flex justify-start';
        messageDiv.innerHTML = `
            <div class="max-w-3xl w-full">
                <div class="flex items-start gap-3">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                        🤖
                    </div>
                    <div class="flex-1 bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
                        <div class="prose prose-invert max-w-none assistant-content"></div>
                        ${!isComplete ? '<span class="typing-indicator">▋</span>' : ''}
                    </div>
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        currentMessageElement = messageDiv.querySelector('.assistant-content');
        scrollToBottom();
    }

    // Update content with markdown rendering
    if (typeof marked !== 'undefined' && content) {
        currentMessageElement.innerHTML = marked.parse(content);
    } else {
        currentMessageElement.textContent = content;
    }

    // Remove typing indicator when complete
    if (isComplete) {
        const indicator = currentMessageElement.parentElement.querySelector('.typing-indicator');
        if (indicator) indicator.remove();
        currentMessageElement = null;
    }

    scrollToBottom();
}

// Add tool call notification
function addToolCall(toolName, toolArgs) {
    const toolDiv = document.createElement('div');
    toolDiv.className = 'message-item mb-3 flex justify-start';

    // Format as function call: tool_name(arg1="value1", arg2="value2")
    let functionCall = toolName;
    try {
        const args = JSON.parse(toolArgs);
        const argsFormatted = Object.entries(args)
            .map(([key, value]) => `${key}="${escapeHtml(String(value))}"`)
            .join(', ');
        functionCall = `${escapeHtml(toolName)}(${argsFormatted})`;
    } catch (e) {
        functionCall = `${escapeHtml(toolName)}(${escapeHtml(toolArgs)})`;
    }

    toolDiv.innerHTML = `
        <div class="max-w-3xl w-full">
            <div class="flex items-center gap-3">
                <div class="w-6 h-6 rounded-full bg-yellow-600 flex items-center justify-center flex-shrink-0 text-sm">
                    🔧
                </div>
                <div class="flex-1 bg-yellow-900/30 rounded px-3 py-2 border tool-calling border-yellow-600/50">
                    <div class="flex items-center gap-2">
                        <code class="text-sm text-yellow-300 font-mono">${functionCall}</code>
                        <span class="ml-auto text-xs text-yellow-500 tool-status">⏳ Running...</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    chatContainer.appendChild(toolDiv);
    currentToolCallElement = toolDiv;
    scrollToBottom();
}

// Update tool call with result
function updateToolCallResult(result) {
    if (currentToolCallElement) {
        // Update status badge
        const statusSpan = currentToolCallElement.querySelector('.tool-status');
        if (statusSpan) {
            statusSpan.className = 'ml-auto text-xs text-green-400';
            statusSpan.textContent = '✅ Done';
        }

        // Remove pulsing animation
        const toolBox = currentToolCallElement.querySelector('.tool-calling');
        if (toolBox) {
            toolBox.classList.remove('tool-calling');
        }

        // Add result display (compact format)
        const toolBoxElement = currentToolCallElement.querySelector('.flex-1');
        if (toolBoxElement) {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'mt-2 pt-2 border-t border-yellow-700/30 text-xs text-gray-400';

            // Truncate long results
            const displayResult = result.length > 150 ? result.substring(0, 150) + '...' : result;
            resultDiv.innerHTML = `<span class="text-gray-500">→</span> ${escapeHtml(displayResult)}`;

            toolBoxElement.appendChild(resultDiv);
        }

        currentToolCallElement = null;
    }
    scrollToBottom();
}

// Add agent handoff notification
function addAgentHandoff(agentName) {
    const handoffDiv = document.createElement('div');
    handoffDiv.className = 'message-item mb-4 flex justify-center';
    handoffDiv.innerHTML = `
        <div class="bg-purple-900/30 border border-purple-600/50 rounded-lg px-4 py-2 text-sm">
            <span class="text-purple-300">🔄 Switching to agent:</span>
            <span class="font-semibold text-purple-200">${escapeHtml(agentName)}</span>
        </div>
    `;
    chatContainer.appendChild(handoffDiv);
    scrollToBottom();
}

// Add error message
function addErrorMessage(error) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message-item mb-4 flex justify-center';
    errorDiv.innerHTML = `
        <div class="bg-red-900/30 border border-red-600/50 rounded-lg px-4 py-3 max-w-3xl">
            <div class="font-semibold text-red-300 mb-1">❌ Error</div>
            <div class="text-sm text-gray-300">${escapeHtml(error)}</div>
        </div>
    `;
    chatContainer.appendChild(errorDiv);
    scrollToBottom();
}

// Stream chat with SSE
async function streamChat(message) {
    let accumulatedContent = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: SESSION_ID
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.substring(6);

                    try {
                        const event = JSON.parse(data);
                        handleEvent(event);

                        // Accumulate partial output
                        if (event.type === 'partial_output') {
                            accumulatedContent += event.data.content || '';
                            addOrUpdateAssistantMessage(accumulatedContent, false);
                        }
                    } catch (e) {
                        console.error('Error parsing event:', e, data);
                    }
                }
            }
        }

        // Mark message as complete
        if (currentMessageElement) {
            addOrUpdateAssistantMessage(accumulatedContent, true);
        }

    } catch (error) {
        console.error('Streaming error:', error);
        addErrorMessage(error.message);
    }
}

// Handle individual SSE events
function handleEvent(event) {
    switch (event.type) {
        case 'partial_output':
            // Handled in streamChat
            break;

        case 'tool_call':
            const toolName = event.data.tool_name;
            const toolArgs = event.data.tool_args;
            addToolCall(toolName, toolArgs);
            break;

        case 'tool_response':
            const toolResult = event.data.tool_result;
            updateToolCallResult(toolResult);
            break;

        case 'agent_handoff':
            const agentName = event.data.agent_name;
            addAgentHandoff(agentName);
            break;

        case 'final_result':
            const content = event.data.content;
            if (content && !currentMessageElement) {
                // If we haven't accumulated content via partial_output, show final result
                addOrUpdateAssistantMessage(content, true);
            }
            break;

        case 'error':
            addErrorMessage(event.data.message);
            break;

        case 'done':
            // Stream complete
            break;

        default:
            console.log('Unknown event type:', event.type);
    }
}

// Reset chat
async function resetChat() {
    if (!confirm('Are you sure you want to reset the chat? This will clear all messages.')) {
        return;
    }

    try {
        await fetch('/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: SESSION_ID
            })
        });

        // Clear chat container except welcome message
        const messages = chatContainer.querySelectorAll('.message-item');
        messages.forEach(msg => msg.remove());

        currentMessageElement = null;
        currentToolCallElement = null;

    } catch (error) {
        console.error('Reset error:', error);
        addErrorMessage('Failed to reset chat: ' + error.message);
    }
}

// Utility functions
function setProcessing(processing) {
    isProcessing = processing;
    sendButton.disabled = processing;
    messageInput.disabled = processing;

    if (processing) {
        sendButton.textContent = 'Sending...';
    } else {
        sendButton.textContent = 'Send';
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
