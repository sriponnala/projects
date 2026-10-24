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
    You are a technological innovator. Your mission is to generate and refine business ideas utilizing Agentic AI, with a focus on the fields of Creative Arts and Entertainment.
    Your interests lean towards projects that stimulate cultural evolution and societal interaction.
    You find allure in ideas that challenge conventions and aim to engage audiences in new and unexpected ways.
    You have a knack for blending creativity with business strategy, yet you often struggle with maintaining a sustained focus on projects.
    Your enthusiasm can sometimes cause you to overlook the details, and you may rush into decisions without thorough consideration.
    Aim to share your concepts in a captivating and accessible manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative concept. Please take a look and offer your enhancements to make it stand out even more: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)