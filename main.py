"""Fashion Chatbot """
import sys

class FilteredStderr:
    """Filter stderr to suppress specific LangChain schema warnings"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, text):
        # Suppress the "Key 'title' is not supported in schema" warning
        if "Key" in text and "is not supported in schema" in text:
            return
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

# Install filtered stderr globally (mimics notebook's %%capture --no-stderr)
sys.stderr = FilteredStderr(sys.stderr)

from langchain_core.messages import HumanMessage, AIMessage
from services import get_vector_store
from graph import build_graph

class FashionChatbot:
    """Shopping assistant with clean output (no logs just chat)"""
    
    def __init__(self, verbose: bool = False):
        
        self.verbose = verbose
        # Load vector stores 
        get_vector_store()
        
        # Build LangGraph
        self.graph = build_graph()
        
        if verbose:
            print("\nFashion Chatbot ready! (Debug Mode)\n")
        else:
             print("\nFashion Chatbot ready!\n")
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Send message and get response with memory persistence"""
        thread = {"configurable": {"thread_id": thread_id}}
        
        response = None
        last_ai_message = None
        
        if self.verbose :
            print(f"thi is the users input query \n {user_message} \n")

        # Stream graph and get response
        for event in self.graph.stream(
                {"messages": [HumanMessage(content=user_message)]},
                thread,
                stream_mode="values"
            ):
                if "messages" in event and event["messages"]:
                    # Get only the last AI message
                    for msg in reversed(event["messages"]):
                        if isinstance(msg, AIMessage):
                            if self.verbose : 
                                print(f"\n{msg}\n")
                            last_ai_message = msg
                            response = msg.content
                            break
        
        
        # Print clean conversation
        print(f"\n{'='*60}")
        
        if last_ai_message and response:
            print(f"Assistant: {response}")
        
        print(f"{'='*60}\n")
        return response
    
    def interactive(self):
        """Interactive chat session"""
        import uuid
        thread_id = str(uuid.uuid4())[:8]  # Unique conversation ID
        
        print("="*60)
        print("  Fashion Chatbot - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Goodbye! Happy shopping!")
                    break
                
                if not user_input:
                    continue
                
                self.chat(user_input, thread_id)
                
            except KeyboardInterrupt:
                print("\n\n Goodbye! Happy shopping!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Run the chatbot"""
    chatbot = FashionChatbot(verbose=False)
    chatbot.interactive()


if __name__ == "__main__":
    main()
