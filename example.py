from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")
# rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', dpo_dir='AmberYifan/llama-7b-sft-DPO')

# rs = RewardSampling(llm_dir='meta-llama/Meta-Llama-3-8B-Instruct', rm_dir='Ray2333/GRM-Llama3-8B-rewardmodel-ft', draft_dir='turboderp/Qwama-0.5B-Instruct')
rs = RewardSampling(llm_dir='lblaoke/qwama-0.5b-skywork-pref-dpo-trl-v2', rm_dir='turboderp/Qwama-0.5B-RewardModel')

# reward: batch_size x 1
# response, _ = rs.generate("How did US states get their names?", max_new_token=64)

chosen = "C# (pronounced \"C sharp\") is a modern, object-oriented programming language developed by Microsoft. It is widely used for building various types of applications, including web applications, desktop applications, mobile applications, and games. C# is similar to other programming languages such as Java and C++, and it is known for its simplicity and ease of use. C# is a powerful language that provides a rich set of libraries and frameworks that make it easy to build robust and scalable applications.\n\nHere is a brief overview of some key features of C#:\n\n1. Object-oriented: C# is an object-oriented language, which means it uses the concept of objects to represent real-world entities and their behavior.\n\n2. Cross-platform: C# can be used to build applications for multiple platforms, including Windows, macOS, and Linux.\n\n3. Strongly typed: C# is a strongly typed language, which means that variables must be declared with a specific type, and their type cannot be changed at runtime.\n\n4. Event-driven: C# uses an event-driven programming model, which means that programs are built around the concept of events, such as user input or network activity.\n\n5. Garbage-collected: C# has a garbage collector that automatically manages memory allocation and deallocation, making it easier to write memory-efficient and robust applications.\n\n6. Community-driven: C# has a large and active community of developers, who contribute to the language and its libraries through open-source projects and other initiatives.\n\nOverall, C# is a versatile and powerful programming language that is widely used for building a variety of applications."
rejected = "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles."

chosen_prob = rs.get_likelihood(chosen)
rejected_prob = rs.get_likelihood(rejected)

# print(response)
print(chosen_prob, rejected_prob)
