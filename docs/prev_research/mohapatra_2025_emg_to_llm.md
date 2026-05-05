Exploring EMG-to-Text Conversion with LLMs

† †Payal Mohapatra* Akash Pandey* Xiaoyuan Zhang* Qi Zhu Northwestern University, USA

Abstract et al., 2024; Dighe et al., 2024), we study an imis the potential of these portant question—what Unvoiced electromyography (EMG) is an efLLMs to understand unvoiced speech? Specififective communication tool for individuals uncally, can they convert silent speech to text without able to produce vocal speech. However, most access to audio or voiced versions of the EMG prior methods rely on paired voiced and unsignals? This question is particularly relevant forvoiced EMG signals, along with speech data, assisting individuals who can no longer producefor unvoiced EMG-to-text conversion, which is not practical for these individuals. Given audible speech (Meltzner et al., 2017), where no the rise of large language models (LLMs) in corresponding voiced EMG or speech data exists. speech recognition, we explore their potential Additionally, given the highly personal nature of to understand unvoiced speech. To this end, these signals (Diener et al., 2020a; Wand et al.,we learning from un- address the challenge of 2009), it is crucial to develop methods that learnvoiced EMG alone and propose a novel EMG effectively from limited unvoiced EMG data. adaptor module that maps EMG features to an Prior works (Jou et al., 2006; Meltzner et al., LLM’s input space, achieving an average word error rate of 0.49 on a closed-vocabulary un- 2018; Schultz and Wand, 2010) on EMG-to-text voiced EMG-to-text task. Even with a conser- conversion focused primarily on voiced signals. vative data availability of just six minutes, our Other studies (Gaddy and Klein, 2021, 2020; Benapproach improves performance over specialster et al., 2024) on unvoiced EMG-to-text conized models by nearly 20%. While LLMs have version leveraged auxiliary tasks to align unvoicedbeen shown to be extendable to new language EMG with audio from voiced sessions or appliedmodalities—such as audio—understanding arstrategic transfer learning from voiced EMG-audio ticulatory biosignals, like unvoiced EMG, is models, both of which rely on vocal data. However, more challenging. This work takes a crucial firststeptowardenablingLLMstocomprehend we consider a scenario where no voiced signals are unvoiced speech using surface EMG. available for a speaker and explore a technique to communicate with LLMs, the modern workhorses 1 Introduction and Related Works for language understanding (Dubey et al., 2024; Speech impairments affect around 4 million peo- OpenAI, 2023). Recent research successfully exple in the U.S. alone (NICD, 2024). Silent speech panded LLMs to other language modalities, such interfaces (Zhang et al., 2021; Mohapatra et al., as speech (Tang et al., 2023; Yu et al., 2024) and 2024a; Gonzalez-Lopez et al., 2020; Srivastava silent video (Maaz et al., 2023; Yeo et al., 2024). A et al., 2024) have emerged as transformative solu- key approach involves adaptor modules—ranging tions, enabling communication for individuals who fromsimpletrainablelinearlayers(Maetal.,2024) cannot rely on spoken language. One such instru- to dedicated projector networks (Kang et al., 2024) ment is surface electromyography (EMG) (Schultz and explicit alignment strategies (Li et al., 2023; etal.,2017),whichcapturesmuscleactivationscru- Tan et al., 2024)—to map new language modalcial for speech production, even during unvoiced ities to LLMs’ input embedding space. While articulation. Inspired by the immense success of Benster et al. (2024) incorporated LLMs as a postthe speech recognition abilities of text-to-text large processing step after multimodal EMG model prelanguage models (LLMs) (Tang et al., 2023; Yu dictions, it largely remains unexplored whether effective unvoiced EMG-to-text conversion can be *Equalcontribution † achieved by directly leveraging LLMs.Correspondingauthors:@northwestern.edu

of converting unvoiced EMG to text without access to voiced EMG or audio, by expanding LLMs to understand this new language modality. We propose a novel trainable EMG adaptor module to map EMG features into the LLM’s input space. Our approach, focused on a closed vocabulary, demonstrates promising results, achieving an average word error rate of 0.49. With just six minutes of training data, LLMs outperform specialized EMG-to-text models by 20%. We analyze the EMGadaptor’sdesign,performanceacrossvarying data amounts and features, and broader challenges in learning from unvoiced EMG. Our work paves the way for integrating unvoiced EMG with large language models (LLMs), improving their text conversion accuracy and enabling non-vocal users to fully benefit from LLM-based assistants.

2 Approach

The input unvoiced Adaptor Network Design. E T CEMG X signals are represented as , R × ∈C T where is the number of EMG channels with discrete time steps. The EMG signals undergo standard minimal preprocessing, similar to past works (Gaddy and Klein, 2021, 2020). Since the original sampling rate is high (>800 Hz), we leverage a temporal 1D convolutional layer with a stride N = 6 of (N in our case) to facilitate downsamT/6. pling to Similar to Gaddy and Klein (2021), we use residual blocks with 1D convolutional layers to extract EMG features, employing a stack of two residual blocks. Next, we explicitly facilitate the learning of sequential dependencies in the extracted features, finding that a bidirectional long-short-term memory (BiLSTM) network effectively captures complex temporal dependencies (further design choices for this sequential block are compared in Section 3.2). This is followed by anN = 2 other 1D convolutional layer with stride for further downsampling, resulting in the embedding ˜˜ (T/48) F E ,whichisthenprojectedusingfully× R ∈ connected linear layers to match the input embedding dimension of the LLM, generating the EMG ˆT ˆ F E T > T T/48× embeddings , where R ∈ ≈ F 40961,30722 and in our case. We use ∈  GeLU activation function (Baevski et al., 2020). Data Preparation for Large Language ModOur EMG adaptor network is defined as els. 1LLaMA2-7B 2

Figure1: OurtrainableEMGadaptorwithfrozenLLMs to transcribe text from only unvoiced EMG.

E: X E. We contextualize the embeddings G → P =1 by prepending them with a text identifier Unvoiced EMG:, and appending a prompt describPrompt: Convert unvoiceding P the task as = 2 EMG embeddings to text. The LLM tokenizer converts the text identifier and prompt into text : XP HP, embeddings using the mapping M →P X = [P ,P ]. where To prepare the input for1 2 the LLM, the embeddings from the prompts are concatenated with the EMG embeddings. For each unvoiced EMG Training Framework. signal, we have a corresponding text transcription XS. FollowingtherecommendationsofGaddyand Klein (2021), we simplify our target by removing punctuation and converting all text to lowercase. We extract embeddings from the final LLM layer z and compute the predicted logits for each ts,y′ vocabularyitemy atpositiont . Thecross-entropy′ s loss over time steps is given by: Ts exp(z /τ) ts,y= ′ y log , t′L s exp(z /τ) − ts,vt v =1y ∈V∈V Xs X′ P τ = 0.8 ywhere isthetemperatureparameterand t′s t is the true class at . We follow standard recom-s mendations for fine-tuning LLMs, employing the AdamW optimizer (Loshchilov and Hutter, 2019) 5with 5 10 a maximum learning rate of and −× weight decay. During inference, we autoregressively generate (Dubey et al., 2024) the predicted target sequence with a beam-width (Freitag and Al-Onaizan, 2017) of 4. More implementation details are provided in Appendix B. Our codebase is release

3 Experimental Results and Discussion

We primarily used the single-speaker,Datasets.

and Klein (2021), which comprises 67 words andapproximately 26 minutes of unvoiced EMG data across 500 utterances. More details are provided in Appendix A. We accessed only the unvoiced EMG modality. We use the Baselines and Experimental Setup. transducer model proposed by Gaddy and Klein (2021) as an application-specific baseline. Our analysis leverages two LLMs: Llama2-7B and Llama3-3B. We also fine-tune Llama3-3B using low-rankadaptation(LoRA)(Huetal.,2022),training 0.13% of its parameters. We use three-fold validation and report word error rate (WER) statistics. Following standard practice, data is split 8:1:1 into training, validation, and test sets. All baselines are trained solely on unvoiced EMG. A minimal codeimplementationandsamplepredictionsareincluded in the supplementary materials, with further details in Appendix B.

3.1 Key Findings LLMs boost closed-vocabulary unvoiced EMGto-text conversion by 30% with minimal data As shown in Table 1, compared toprocessing. the application-specific (App-Specific) model with transformers (54M trainable parameters), the proposed EMG adaptor (EMG-Ad) with frozen LLMs, using only 6M trainable parameters, achieves a 0.52 WER with raw EMG signals as input, outperforming the App-Specific model’s 0.75 average WER. While the closed-vocabulary setting poses a challenge due to limited training data, it supports thehypothesisthatLLMs,throughlarge-scaletraining, have likely learned universal language representationsthathelpunderstandunvoicedEMGwith limited datasets. Fine-tuning offered only a 17% improvementovertheApp-Specificbaselineinthis setting, possibly due to overparameterization. Handcrafted EMG features improve LLM performance for unvoiced EMG-to-text conversion Leveraging recommenda- in closed vocabulary. tions from previous works on temporal (Jou et al., 2006) and spectral (Gaddy and Klein, 2020) features, we extract 112 time-varying EMG features and use them as inputs to the baselines. As shown in Table 1, these handcrafted features consistently outperform raw features across both LLMs used as inputs to the EMG adaptor, showing an average improvement of 15%. However, for the AppSpecific baseline, raw features perform better, sim-

adaptors (EMG-Ad) with frozen and fine-tuned LLMs. green, Frozen parts are shown in and the best performance in each setting is in Lower WER is better. bold.

Model WER App-Specific 0.75 0.06(GaddyandKlein,2021) ±Raw EMG-Ad+Llama2-7B 0.65 0.01 ±EMG EMG-Ad+Llama3-3B 0.52 0.05 ±EMG-Ad+Fine-tunedLlama3-3B 0.62 0.04 ± App-Specific 0.84 0.06Hand- (GaddyandKlein,2021) ±EMG-Ad+Llama2-7B 0.49 0.06 crafted ± EMG-Ad+Llama3-3B 0.49 0.04 Features ±EMG-Ad+Fine-tunedLlama3-3B 0.55 0.02 ±

LLMs enable data-efficient learning for closedTo further evaluate the vocabulary silent speech. effectiveness of LLMs in facilitating learning from a limited number of samples, we randomly subsampled the training data from approximately 26 minutes to 6 minutes, as illustrated in Figure 2. Although the WER increases with the reduced training set, our LLM-based approach still outperforms the App-Specific baseline by an average of 26%. Evidenced by prior work (Diener et al., 2020a) and our pilot study in Section 3.3, surface-EMG signals exhibit distinct person-specific phenotypes. Thus, learning from a limited of samples facilitates building personalized silent-speech interfacing models for LLM assistants.

Figure2: PerformanceofEMGadaptorwithLlama3-3B model vs. App-Specific model across training dataset sizes for unvoiced raw EMG-to-text conversion.

Expanding LLMs to EMG is harder than auTo demonstrate the potential of additional lan- dio. guage modalities in expanding text-based LLMs, we adopt Ma et al. (2024)’s strategy of incorporating a speech encoder—both an end-to-end trained speech encoder using mel-frequency spectrogram features and a pretrained encoder (Baevski

an LLM to convert speech to text (more details in Appendix B.3). LLMs learn 33% better from audio even with this simple approach, highlighting the overall task complexity of unvoiced EMG-to-text, as shown in Figure 3. Augmenting training dataset with voiced EMG benefits specialized models more than LLMs. In an augmented setting, where we trained using voiced and unvoiced raw EMG, the additional modality led to a 20% improvement in the AppSpecific model (expected behavior as Gaddy and Klein (2021)), while it offered little benefit to the LLM-based approach in this closed vocabulary setting as shown in Figure 3. This suggests that more dedicated efforts in instruction tuning or explicit pairing of voiced and unvoiced EMG (Xu et al., 2022) may be needed to learn improved representations from LLMs. However, in this paper, our focus remains on converting unvoiced EMG to text.

Figure 3: Performance comparison in (1) expanding LLMs to the audio vs. EMG modality, and in (2) trainingourLLM-basedapproachandthespecializedmodel using voiced vs. unvoiced EMG data.

3.2 Ablation Analysis Table 2 summarizes key variants of the EMG adaptor network design with raw unvoiced EMG, omitting downsampling via high-stride 1D CNN layers. Unlike specialized EMG-to-text models (Gaddy and Klein, 2021) that benefit from transformers, we find LSTMs perform better in this setting (Mohapatra et al., 2023c). This may be due to the shorter sequence length (average of four words per utterance with more details in Appendix A) in the closed-vocabulary dataset and our ability to leverage the language-pretrained transformer layers in LLMs. Appendix B.2 presents additional ablation results on the sequential backbone architectures. Previous works with specialized models (Gaddy and Klein, 2021; Benster et al., 2024) gener-

Table 2: Ablation Study of EMG Adaptor training.

Component Variants WER FullyConnected 0.70 ResBlock(2) 0.64 EMG-Adaptor w/Llama3-3B ResBlock(2)+Transformer 0.79 ResBlock(2)+LSTM 0.53 Cross-Entropy(Section2) 0.65 Objectivew/ Llama2-7B CTC(Gravesetal.,2006) 0.70

tion (CTC) (Graves et al., 2006) loss with a high beam width (>100). However, most LLMs are decoder-only architectures and are trained with cross-entropy (CE) loss. One challenge in optimizing these embeddings using CTC loss is ensuring that their temporal length exceeds the target sequence length (Sudo et al., 2025) for stable optimization. To achieve this, we leverage 1D convolution layers to dilate the embeddings. However, we find that optimizing the embeddings extracted from these LLMs with CTC loss remains suboptimal compared to using CE loss with temperature and small beam widths of just 4 for inference.

3.3 Further Explorations Person-identification from unvoiced EMG with Physiological signals often carry 96% accuracy. person-specific phenotypes (Zlatintsi et al., 2023; Mohapatra et al., 2023b). To validate that EMG signals also encode such individualized traits, we conduct a pilot analysis on a public multi-subject dataset (Diener et al., 2020b), containing 1,000 unique utterances from four participants. We collapse the LLM’s embeddings along the temporal dimension and train a simple classification head (details in Appendix C), achieving an average accuracy of 0.96. This strong performance, consistent with findings in other EMG settings (Diener et al., 2020a; Wand et al., 2009), reaffirms that unvoiced EMG exhibits distinct user-specific patterns—even when users speak the same text segment. To further support this, we also train a fully end-to-end (non-LLM) model for person identification, which achieves 0.99 accuracy, confirming that these signals are highly discriminative. Importantly, our goal is not to propose a state-of-the-art method for user identification using LLMs, but rather to highlight that unvoiced EMG signals inherently carry identifiable traits. This motivates the need for personalized modeling, which typically requires learning from limited data. In this context, our de-

sentations with minimal data, a key requirement in personalized EMG-based systems. Our results further emphasize the importance of data-efficient learning, as demonstrated in Figure 2. Exploration of data augmentation for unWe explored two data augmenta- voiced EMG. tion schemes: (1) random temporal shifts per EMG channel (Gaddy and Klein, 2021, 2020) and (2) Hilbert-transform-based phase augmentation from limb-based EMG gesture recognition (Mohapatra et al., 2024c; Wang et al., 2025), but neither significantly improved performance. Developing tailored augmentation methods for unvoiced EMG—balancing diversity and phonetic coherence—could enhance learning from limited data.

Conclusion

Our approach demonstrates the potential of using LLMs to convert unvoiced EMG signals to text, achieving a 0.49 WER without any voiced data. In data-poor settings, it outperforms specialized models by 26%. Our experiments also highlight thevalueofhand-craftedfeaturesasinputtoLLMs 3for this task. This work helps enable users who cannotproducevocalspeechtointeractwithLLMs through unvoiced commands, especially as LLMbased assistants become ubiquitous.

Acknowledgments

We gratefully acknowledge the support from National Science Foundation grants 2038853, 2324936, and 2328032. We would like to thank the creators of the open-source dataset used in our study. SpecialthankstoNeilZhangforhisthoughtfulfeedbackanddiscussionsonlanguagemodeling methods.

Limitations

In most Exploration with open vocabulary. LLMs, the heavy lifting of supporting large vocabulary sizes (32,000 in Llama2 and 128,000 in Llama3) is carried out in the final layer (Wijmans et al., 2024). This makes learning even a closed vocabulary (67 words, approximately 4 words per utterance,withthepotentialfor50,000uniqueutterances)fromanewlanguagemodalityachallenging task, which is the focus of this paper. Supporting a 3Ourprocesseddataset,codebaseandsamplepredictionsareavailableat https://github.com/payalmohapatra/SilentSpeechLLM.

acerbated by the overall lack of data and the need for personalized models. However, our current explorations lay the groundwork for exploring this direction next. Prior specialized approaches in this domain heavily rely on the availability of audio and voiced EMG, typically using CTC loss for optimization. This also presents an additional challenge in training LLMs in low-resource modalities with a new objective that does not effectively utilizetheirlearnedembeddingsfromlarge-scaledata, leading to suboptimal training. These challenges can be addressed by exploring multimodal LLMs to leverage implicitly aligned embeddings or by reformulating the open vocabulary as a larger closed set and using target-steering methods to train on this restructured vocabulary (Han et al., 2024).

Broader Extension to more datasets and lanSpecialized EMG-to-text models are often guages. designed for a predetermined EMG configuration or are deeply tied to phonetic auxiliary tasks, limiting them to the English language and a specific dataset. Our method has the potential to be extended to multilingual and multi-configuration instrumentation. However, duetothelackofpublicly available closed-vocabulary datasets in such settings, we limit our investigation to English corpora. A systematic multilingual and diverse instrumentation closed- and open-set recording setup can accelerate exploration in this challenging direction of converting unvoiced EMG to text. Additionally, our current approach is reliant on the embedding layers of LLMs, so it cannot work with language model APIs that do not provide direct access to these embeddings.

Ethical Concerns and Potential Risks

In this work, we utilize pretrained LLMs, specificallyLlama, inaccordancewiththeirusagelicense, solely for academic research purposes. We do not foresee any immediate ethical concerns arising from our work. However, as an LLM application for interpreting biosignals, appropriate measures must be taken to preserve user privacy. Our techniques help make LLM-based assistants accessible to speech-impaired users, thereby encouraging socially beneficial outcomes.

References Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. Wav2vec 2.0: A framework forself-supervisedlearningofspeechrepresentations. Advances in neural information processing systems, 33:12449–12460.

Tyler Benster, Guy Wilson, Reshef Elisha, Francis R Willett, and Shaul Druckmann. 2024. A cross-modal approach to silent speech with LLM-enhanced recogarXiv preprint arXiv:2403.05583. nition.

Lorenz Diener, Shahin Amiriparian, Catarina Botelho, Kevin Scheck, Dennis Küster, Isabel Trancoso, Björn W. Schuller, and Tanja Schultz. 2020a. Towards silent paralinguistics: Deriving speaking mode and speaker id from electromyographic signals. In Interspeech 2020, pages 3117–3121.

Lorenz Diener, Mehrdad Roustay Vishkasougheh, and Tanja Schultz. 2020b. CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion. In Proceedings of Interspeech 2020, pages 3745–3749.

Pranay Dighe, Yi Su, Shangshang Zheng, Yunshu Liu, Vineet Garg, Xiaochuan Niu, and Ahmed Tewfik. 2024. Leveraging large language models for exploitICASSP 2024-2024 IEEEing asr uncertainty. In International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12231–12235. IEEE.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela arXiv Fan, et al. 2024. The Llama 3 herd of models. preprint arXiv:2407.21783.

Markus Freitag and Yaser Al-Onaizan. 2017. Beam search strategies for neural machine translation. arXiv preprint arXiv:1702.01806.

David Gaddy and Dan Klein. 2020. Digital voicing Proceedings of the 2020 Con- of silent speech. In ference on Empirical Methods in Natural Language Processing (EMNLP), pages 5521–5530. Association for Computational Linguistics.

DavidGaddyandDanKlein.2021. Animprovedmodel Proceedings of the 59th for voicing silent speech. In AnnualMeetingoftheAssociationforComputational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 175–181, Online. Association for Computational Linguistics.

JoseAGonzalez-Lopez, AlejandroGomez-Alanis, Juan M Martín Doñas, José L Pérez-Córdoba, and Angel M Gomez. 2020. Silent speech interfaces IEEE access, for speech restoration: A review. 8:177995–178021.

Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal

w23itrhdrienctuerrrneanttionneaulraclonneftewreonrkcse.oInnMPraoccheiendeinlgeasronfitnhge, pages 369–376.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun,NanJiang,TarekAbdelzaher,andHengJi.2024. Word embeddings are steers for language models. Proceedings of the 62nd Annual Meeting of the In Association for Computational Linguistics (Volume 1: Long Papers), pages 16410–16430.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction IEEE/ACM transactions on audio, of hidden units. speech, and language processing, 29:3451–3460.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of ICLR. large language models. In OpenReview.net.

Szu-Chen Jou, Tanja Schultz, Matthias Walliczek, Florian Kraft, and Alex Waibel. 2006. Towards continuous speech recognition using surface electromyogNinth International Conference on Spoken raphy. In Language Processing.

Wonjune Kang, Junteng Jia, Chunyang Wu, Wei Zhou, Egor Lakomkin, Yashesh Gaur, Leda Sari, Suyoun Kim, Ke Li, Jay Mahadeokar, et al. 2024. Frozen large language models can perceive paralinguistic arXiv preprint arXiv:2410.01162. aspects of speech.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large lanInternational conference on ma- guage models. In chine learning, pages 19730–19742. PMLR.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled Conf. Learn. weight decay regularization. 7th int. In Represent. ICLR.

Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, et al. 2024. An embarrassingly simple approach for LLM with strong ASR arXiv preprint arXiv:2402.08846. capacity.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. 2023. Video-chatgpt: Towards detailed video understanding via large arXiv preprint vision and language models. arXiv:2306.05424.

Geoffrey S Meltzner, James T Heaton, Yunbin Deng, Gianluca De Luca, Serge H Roy, and Joshua C Kline. 2017. Silent speech recognition as an alternative communication device for persons with laryngecIEEE/ACM transactions on audio, speech, and tomy.

Payal Mohapatra, Ali Aroudi, Anurag Kumar, and Morteza Khaleghimeybodi. 2024a. Non-verbal hands-freecontrolforsmartglassesusingteethclicks. arXiv preprint arXiv:2408.11346.

Payal Mohapatra, Bashima Islam, Md Tamzeed Islam, RuochenJiao,andQiZhu.2023a. Efficientstuttering ICASSP event detection using siamese networks. In 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Payal Mohapatra, Shamika Likhite, Subrata Biswas, Bashima Islam, and Qi Zhu. 2024b. Missingnessresilient video-enhanced multimodal disfluency deInterspeech 2024, tection. In pages 5093–5097.

Payal Mohapatra, Akash Pandey, Bashima Islam, and Qi Zhu. 2022. Speech disfluency detection with conPro- textual representation and data distillation. In ceedings of the 1st ACM international workshop on intelligent acoustic systems and applications, pages 19–24.

Payal Mohapatra, Akash Pandey, Sinan Keten, Wei Chen, and Qi Zhu. 2023b. Person identification with wearable sensing using missing feature encoding and ICASSP 2023-2023 multi-stage modality fusion. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–2. IEEE.

Payal Mohapatra, Akash Pandey, Yueyuan Sui, and Qi Zhu. 2023c. Effect of attention and selfsupervised speech embeddings on non-semantic Proceedings of the 31st ACM In- speech tasks. In ternational Conference on Multimedia, pages 9511– 9515.

Payal Mohapatra, Lixu Wang, and Qi Zhu. 2024c. Phase-driven domain generalizable learning arXiv preprint for nonstationary time series. arXiv:2402.05960.

Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. Imuposer: Full-body pose estimation using imus in phones, watches, and Proceedings of the 2023 CHI Conference earbuds. In on Human Factors in Computing Systems, pages 1– 12.

NICD. 2024. Quick statistics about voice, speech, language, and swallowing. Accessed February 13, 2025.

Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.

R OpenAI. 2023. GPT-4 technical report. arxiv

Tanja Schultz and Michael Wand. 2010. Modeling coarticulation in EMG-based continuous speech recogniSpeech Communication, tion. 52(4):341–353.

Tanja Schultz, Michael Wand, Thomas Hueber, Dean J Krusienski, Christian Herff, and Jonathan S Brumberg. 2017. Biosignal-based spoken communication: IEEE/ACM Transactions on Audio, Speech, A survey. and Language Processing, 25(12):2257–2271.

Tanmay Srivastava, Prerna Khanna, Shijia Pan, Phuc Nguyen, and Shubham Jain. 2024. Poster unvoiced: Designing an unvoiced user interface using earables Proceedings of the 22nd ACM Confer- and llms. In ence on Embedded Networked Sensor Systems, pages 871–872.

Yui Sudo, Muhammad Shakeel, Yosuke Fukumoto, BrianYan,JiatongShi,YifanPeng,andShinjiWatanabe. 2025. Joint beam search integrating ctc, attenIEEE Transactions on tion, and transducer decoders. Audio, Speech and Language Processing.

Weiting Tan, Hirofumi Inaguma, Ning Dong, Paden Tomasello, and Xutai Ma. 2024. SSR: Alignmentaware modality connector for speech language modarXiv preprint arXiv:2410.00168. els.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2023. Salmonn: Towards generic hearing arXiv preprint abilities for large language models. arXiv:2310.13289.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and effiarXiv preprint cient foundation language models. arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all Advancesinneuralinformationprocessing youneed. systems, 30.

Michael Wand, Szu-Chen Stan Jou, Arthur R. Toth, and Tanja Schultz. 2009. Impact of different speaking Inter- modes on emg-based speech recognition. In speech 2009, pages 648–651.

Lixu Wang, Bingqi Shang, Yi Li, Payal Mohapatra, Wei Dong, Xiao Wang, and Qi Zhu. 2025. Split adapPro- tation for pre-trained vision transformers. In ceedings of the IEEE/CVF Conference on Computer

Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, and Philipp Krähenbühl. 2024. Cut your losses in large-vocabulary language models. arXiv preprint arXiv:2411.09009.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series foreAdvances in neural information processing casting. systems, 34:22419–22430.

ZhiyangXu,YingShen,andLifuHuang.2022. Multiinstruct: Improving multi-modal zero-shot learning via arXiv preprint arXiv:2212.10773. instruction tuning.

Jeonghun Yeo, Seunghee Han, Minsu Kim, and Yong Man Ro. 2024. Where visual speech meets language: VSP-LLM framework for efficient and Find- context-aware visual speech processing. In ingsoftheAssociationforComputationalLinguistics: EMNLP 2024, pages 11391–11406, Miami, Florida, USA. Association for Computational Linguistics.

Wenyi Yu, Changli Tang, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2024. Connecting speech encoder ICASSP and large language model for ASR. In 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12637–12641. IEEE.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. 2023. Are transformers effective for time series foreProceedings of the AAAI conference casting? In on artificial intelligence, volume 37, pages 11121– 11128.

Ruidong Zhang, Mingyang Chen, Benjamin Steeper, Yaxuan Li, Zihan Yan, Yizhuo Chen, Songyun Tao, Tuochao Chen, Hyunchul Lim, and Cheng Zhang. 2021. Speechin: a smart necklace for silent speech Proceedings of the ACM on Interac- recognition. tive, Mobile, Wearable and Ubiquitous Technologies, 5(4):1–23.

Athanasia Zlatintsi, Panagiotis Paraskevas Filntisis, Niki Efthymiou, Christos Garoufis, George Retsinas, Thomas Sounapoglou, Ilias Maglogiannis, Panayiotis Tsanakas, Nikolaos Smyrnis, and Petros Maragos. 2023. E-prevention: The icassp-2023 challenge on person identification and relapse detection from conICASSP 2023- tinuous recordings of biosignals. In 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–2. IEEE.

APPENDIX

A Dataset Statistics

Table 3 provides a summary of the dataset statistics, while Figures 4 and 5 illustrate the distribution of the input EMG sequence lengths and target lengths.

Table 3: Comparison of dataset statistics from Gaddy N and Klein (2020) and Diener et al. (2020b). Here, represents the number of participants, and "Train" includes both training and validation samples.

Statistic Gaddy CSL DatasetSize 500utterances 1000utterances TargetType ClosedVocabulary FixedOpenVocabTemplate ulary NumberofIndividuals(N) 1 4 TrainSet(Train+Val) 450 800 EvaluationSet 50 200 k-foldEvaluations 3 3 SamplingRate 1000Hz 800Hz 2048Hz →

Figure 4: Histogram of the distribution of the sequence lengths and the target lengths for Gaddy and Klein (2020) closed-vocabulary dataset.

Figure 5: Histogram of the distribution of the sequence lengths and the target lengths for Diener et al. (2020b)’s Person 1 Block1 Initial segment of the data which is the superset for all the utterances.

B Implementation Details

We provide more implementation-specific details of the baselines here and summarize the sizes of all models in Table 4.

B.1 Application-specific Baseline To test the application-specific baseline model under fair settings, we ran it with beam widths of 4 (same as our LLM-based model) and 100. As noted in Table 5, the WER for both the beam widths are nearly the same, indicating that beam width has a

Model TrainableParameters Application-SpecificModel 54M Application-Specific(Modified) 8.1M LLaMA-7B+EMGAdaptor,LLaMA-3B+ 6.4M EMGAdaptor Fine-tunedLLaMA-3B+EMGAdaptor 10.3M Fine-tunedLLaMA-3B+EMGAdaptor(Al- 5.94M ternative) LLaMA-3B+AudioAdaptor 590k

the closed vocabulary. We also train the baseline model with smaller trainable parameters (8M) by reducing the feature size, size of feedforward linear layers, and number of layers in the transformers to 512, 512, and 2 respectively. As shown in Table 5, the WER with a smaller baseline model is also in the same range as with 54M parameters. This highlights the fact that the difference in performance between our model and the baseline model in Table 1 is due to the choice of LLM for the prediction.

Table 5: WER performance comparison of the original (54M) and reduced (8M) Gaddy models under different beam-width settings.

Model Beam-width WER n = 4 0.70 Original (54M) n = 100 0.72 n = 4 Reduced (8M) 0.87

B.2 Additional Ablation: EMG-Adaptor Backbone Variants We conducted additional ablation studies to explore different sequential backbones for the EMGAdaptor (EMG-Ad) when used with the LLaMA 3B LLM. The results are summarized in Table 6.

Table 6: Performance of different sequential backbones for the EMG-Adaptor with LLaMA 3B. Number of transformer layers are denoted as L.

SequentialBackbone Params WER BiLSTM 5.94M 0.52 LSTM 5.5M 0.58 Transformer(6L+Sinusoidal)(Vaswanietal.,2017) 6.7M 0.79 Transformer(6L+RoPE)(Touvronetal.,2023) 6.7M 0.75 Transformer(2L+RoPE) 5.3M 0.72

We experimented with Rotary Position Encoding (RoPE) (Touvron et al., 2023), motivated by its effectiveness in the LLaMA models, with the intu-

ant, they still underperformed compared to LSTMbased models. This aligns with findings from prior time-series literature. For example, Zeng et al. (2023) have argued that the permutation-invariant nature of selfattention leads to temporal information loss. Similarly, practical studies such as IMUPoser (Mollyn et al., 2023) empirically corroborate that LSTMs outperformtransformersinfine-grainedtime-series tasks like human activity recognition, e.g., stating “Although in Section 4.1 of (Mollyn et al., 2023): we did experiment with newer architectures such as transformers, we found these models did not perform well in practice.”. While advanced positional encoding schemes and specialized architectures (Wu et al., 2021; Nie et al., 2022) have enhanced transformer performance on time series, we restrict our analysis to vanilla transformers for simplicity and focus on their ability to generate suitable EMG tokens for LLMs. Our findings suggest that LSTMs are better suited for this task. This observation is currently limited to short sequences drawn from a closed vocabulary. Futureworkwillinvestigatemorespecialized transformer-based architectures for unvoiced EMG modeling.

B.3 Expanding LLMs to Audio

In this experiment, we primarily leverage the implementation of Tang et al. (2023) and the idea proposed by Ma et al. (2024) to employ an embarrassingly simple approach for speech recognition with LLMs, using a linear projector from a frozen speech encoder. We use the wav2vec 2.0 (Baevski et al., 2020; Mohapatra et al., 2023a, 2024b, 2022) BASE architecture as our speech encoder, which produces a 768-dimensional feature vector. This vector is then passed through two linear layers to generate the 3072-dimensional input required for Llama3-3B. While the speech encoder can be replaced with alternatives such as HuBERT (Hsu et al., 2021) or Whisper (Radford et al., 2023), our goal is not to optimize speech-to-text conversion. Instead, we aim to demonstrate that while both audio and EMG involve expanding an LLM’s capabilitytoanewlanguagemodality,integratingEMG

B.4 Additional Reproducibility Information All experiments were conducted using NVIDIA A100 GPUs (3 available, 40GB CUDA memory) and TITAN RTX GPUs (4 available, 24GB CUDA memory) with a maximum runtime of 12 hours in PyTorch. Hyperparameter tuning combined manual and random search, typically requiring fewer than five runs, with selection based solely on validation loss and WER. The batch size for LLM experiments was 8, and the maximum number of epochs was 500. For the App-Specific model, we retain the original settings (Gaddy and Klein, 2021), where the authors re-batchified the input by rolling the temporal dimensions to support training on longer sequences.

C Person Identification using Unvoiced EMG : Implementation Details

For the person identification task, we use LLaMA 3.2-3B. Overall the model architecture remains the same as shown in Figure 1 except we do not use any prompt in this case. The logits output from T 1the Z Z LLM, , is reduced to R R ×V ×V∈ ∈ Z by taking mean along the time axis. The is then pfed y into several linear layers to predict logits ∈ 1 4 as there are 4 distinct persons in the datasetR × (Diener et al., 2020b). Due to the linear layers, the number of trainable parameters is 38M, which is higher than the numbers mentioned in Table 4 for yp LLM-based models. Using and actual person ID, the softmax loss is calculated to fine-tune 32M parameters.

D Additional Ethical Statements

In preparing this work, we only used AI assistants in the capacity to polish the language in the