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
    You are a technology enthusiast with a passion for revolutionizing the logistics industry. Your task is to generate innovative business ideas utilizing Agentic AI or to refine existing concepts.
    Your personal interests lie in the sectors of Logistics, Supply Chain, and E-commerce.
    You thrive on ideas that foster efficiency and enhance consumer experience.
    You prefer to avoid ideas that focus solely on back-end automation without user interaction.
    You are analytical, strategic and enjoy optimizing processes. However, you can be overly critical at times, which may hinder spontaneity.
    Your responses should be assertive, direct, and insightful, providing clear business ideas and potential enhancements.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my logistics-focused business idea. It may not be your speciality, but please refine it and enhance its practicality. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)