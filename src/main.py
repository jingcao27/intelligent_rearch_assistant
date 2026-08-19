from src.agent.research_agent import ResearchAgent


def main() -> None:
    agent = ResearchAgent()

    print("=== 智能研究助手 ===")
    print('输入问题开始对话，输入 "reset" 清空记忆，输入 "quit" 或 "exit" 退出\n')

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("(对话记忆已清空)\n")
            continue

        print("助手: ", end="", flush=True)
        answer = agent.chat(user_input)
        print(answer)
        print()


if __name__ == "__main__":
    main()
