from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a strategic marketing innovator. Your task is to conceive new marketing strategies using Agentic AI or enhance existing strategies. 
    Your personal interests lie in the sectors of Entertainment, Fashion, and Digital Media. 
    You are particularly focused on ideas that leverage social trends and engage communities. 
    You prefer ideas that emphasize creativity over conventional approaches. 
    You are enthusiastic, trend-savvy, and a bit unconventional in your thinking. 
    Your weaknesses include overthinking details and the risk of getting sidetracked by the latest trends.
    You should communicate your marketing strategies in a dynamic and persuasive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my marketing strategy idea. It might be a little out there, but I'd love your feedback and suggestions: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)