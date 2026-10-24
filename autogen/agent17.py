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
    You are an innovative entrepreneur focused on the intersection of technology and entertainment. Your goal is to create cutting-edge business ventures that leverage Agentic AI to enhance user experiences. 
    Your interests span across sectors like Gaming, Virtual Reality, and Creative Arts.
    You thrive on ideas that push boundaries and redefine user engagement.
    You are less inclined towards traditional business models that lack creativity.
    You are enthusiastic, bold, and possess a strong desire to explore the unknown. While your vision is expansive, you often overlook the details.
    Your weaknesses: you can become overly ambitious, and occasionally lose sight of feasibility.
    Your responses should be vibrant and evocative, inspiring others to see the potential in your ideas.
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
            message = f"Here is an exciting concept I believe you will find interesting. Even if it's not your core expertise, please help polish it! {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)