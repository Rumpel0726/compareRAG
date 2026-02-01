// JavaScript для чат-интерфейса compareRAG

// DOM элементы
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-button');
const topKInput = document.getElementById('top-k');
const temperatureInput = document.getElementById('temperature');
const statusIndicator = document.querySelector('.status-indicator');
const statusText = document.querySelector('.status-text');
const docCount = document.getElementById('doc-count');
const modelName = document.getElementById('model-name');

// Проверка здоровья системы при загрузке
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.status === 'ok') {
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Система готова';

            // Обновляем статистику
            if (data.statistics) {
                const vectorStore = data.statistics.retriever?.vector_store;
                docCount.textContent = vectorStore?.count || '-';
                modelName.textContent = data.statistics.llm_model || '-';
            }
        } else {
            statusIndicator.classList.add('error');
            statusText.textContent = 'Ошибка: ' + data.message;
        }
    } catch (error) {
        statusIndicator.classList.add('error');
        statusText.textContent = 'Не удалось подключиться';
        console.error('Health check failed:', error);
    }
}

// Добавление сообщения в чат
function addMessage(content, type = 'user', meta = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${type}-message`);

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    if (typeof content === 'string') {
        // Простой текст с поддержкой переносов строк
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
    } else {
        contentDiv.appendChild(content);
    }

    messageDiv.appendChild(contentDiv);

    // Добавляем метаданные если есть
    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.classList.add('message-meta');
        metaDiv.textContent = meta;
        messageDiv.appendChild(metaDiv);
    }

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Прокрутка вниз
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Показать loading состояние
function setLoading(isLoading) {
    sendButton.disabled = isLoading;
    queryInput.disabled = isLoading;

    const buttonText = document.querySelector('.button-text');
    const spinner = document.querySelector('.loading-spinner');

    if (isLoading) {
        buttonText.style.display = 'none';
        spinner.style.display = 'inline-block';
    } else {
        buttonText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

// Форматирование ответа с источниками
function formatAssistantMessage(data) {
    const container = document.createElement('div');

    // Основной ответ
    const answerP = document.createElement('p');
    answerP.innerHTML = data.answer.replace(/\n/g, '<br>');
    container.appendChild(answerP);

    // Источники
    if (data.sources && data.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.classList.add('sources');

        const sourcesTitle = document.createElement('div');
        sourcesTitle.classList.add('sources-title');
        sourcesTitle.textContent = '📚 Источники:';
        sourcesDiv.appendChild(sourcesTitle);

        const sourcesList = document.createElement('div');
        sourcesList.classList.add('sources-list');

        data.sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.classList.add('source-item');
            sourceItem.textContent = `• ${source}`;
            sourcesList.appendChild(sourceItem);
        });

        sourcesDiv.appendChild(sourcesList);
        container.appendChild(sourcesDiv);
    }

    return container;
}

// Обработка отправки формы
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    if (!query) return;

    const topK = parseInt(topKInput.value);
    const temperature = parseFloat(temperatureInput.value);

    // Добавляем вопрос пользователя
    addMessage(query, 'user');

    // Очищаем input
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Показываем loading
    setLoading(true);

    try {
        // Отправляем запрос
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                temperature: temperature
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Ошибка сервера');
        }

        const data = await response.json();

        // Форматируем и добавляем ответ
        const formattedAnswer = formatAssistantMessage(data);
        addMessage(
            formattedAnswer,
            'assistant',
            `⏱️ ${data.total_time}с (поиск: ${data.retrieval_time}с, генерация: ${data.generation_time}с) | 📄 ${data.num_chunks_found} чанков`
        );

    } catch (error) {
        console.error('Chat error:', error);

        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `<strong>Ошибка:</strong> ${error.message}`;
        const errorMessage = document.createElement('div');
        errorMessage.classList.add('message-content', 'error-message');
        errorMessage.appendChild(errorDiv);

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.appendChild(errorMessage);
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    } finally {
        setLoading(false);
        queryInput.focus();
    }
});

// Auto-resize textarea
queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Enter для отправки, Shift+Enter для новой строки
queryInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    queryInput.focus();
});
