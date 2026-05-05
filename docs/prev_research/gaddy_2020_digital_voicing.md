David Gaddy Dan Klein and

University of California, Berkeley @berkeley.edu

Abstract

In this paper, we consider the task of digitally voicing silent speech, where silently mouthed0 2 words are converted to audible speech based 0 on electromyography (EMG) sensor measure2 ments that capture muscle impulses. While t prior work has focused on training speechc synthesis models from EMG collected dur- O vocalized ing speech, we are the ﬁrst to train 6 from EMG collected during silently articulated speech. We introduce a method of train] S ing on silent EMG by transferring audio tarA gets from vocalized to silent signals. Our method greatly improves intelligibility of au- . s dio generated from silent EMG compared tos Figure 1: Electromyography (EMG) electrodes placed e a baseline that only trains with vocalized data, on the face can detect muscle movements from speeche decreasing transcription word error rate from[ articulators. 64% to 4% in one data condition and 88% to 1 68% in another. To spur further development v onthistask,weshareournewdatasetofsilent0 people who are no longer able to produce audible and vocalized facial EMG measurements.6 speech, such as individuals whose larynx has been 9 removed due to trauma or disease (Meltzner et al.,2 1 Introduction 0 2017). In addition to these direct uses of digital . voicing for silent speech, it may also be useful as a In this paper, we are interested in in enabling 0 1 component technology for creating silent speechspeech-like communication without requiring 0 to-text systems (Schultz and Wand, 2010), makingsound to be produced. By using muscular sen2 silent speech accessible to our devices and digi- : sor measurements of speech articulator movement, v tal assistants by leveraging existing high-qualityi we aim to capture silent speech - utterances that X audio-based speech-to-text systems. have been articulated without producing sound. In r dig- particular, we focus on the task which we call To capture information about articulator move- a ital voicing, or generating synthetic speech to be ment, we make use of surface electromyography transmitted or played back. (EMG).SurfaceEMGuseselectrodesplacedontop

Digitally voicing silent speech has a wide array of the skin to measure electrical potentials caused of potential applications. For example, it could be by nearby muscle activity. By placing electrodes used to create a device analogous to a Bluetooth around the face and neck, we are able to capture headsetthatallowspeopletocarryonphoneconver- signalsfrommusclesinthespeecharticulators. Figsationswithoutdisruptingthosearoundthem. Such ure 1 shows the EMG electrodes used to capture a device could also be useful in settings where the signals, and Figure 2 shows an example of EMG environment is too loud to capture audible speech signals captured. We collect EMG measurements or where maintaining silence is important. Alter- during both vocalized speech (normal speech pronatively, the technology could be used by some ductionthathasvoicing,frication,andotherspeech

A -audiofromvocalizedspeech V

E EV -EMGfromvocalizedspeech -EMGfromsilentspeech S

AFigure 2: The three components of our data that we will use in our model. The vocalized speech signals and V E E are collected simultaneously and so are time-aligned, while the silent signal is a separate recording of the V S same utterance without vocalization. During training we use all three signals, and during testing we are given just E , from which we must generate audio. Colors represent different electrodes in the EMG data. Note that the S E E silent EMG signal is qualitatively different from its vocalized counterpart . Not pictured, but also included S V inourdataaretheutterancetexts,inthiscase: “Itispossiblethattheinfusoriaunderthemicroscopedothesame.” The War of the Worlds). (from H.G. Well’s

sounds) and silent speech (speech-like articulations on silent EMGE rather than onlyvocalized EMGS E which do not produce sound). We denote these . Training with silent EMG is more challengingV E E EMG signals and , respectively. During the than with vocalized EMG, because when trainingV S A vocalized speech we can also record audio , but on vocalized EMG data we have both EMG inputsV during silent speech there is no meaningful audio andtime-alignedspeechtargets,butforsilentEMG to record. anyrecordedaudiowillbesilent. Oursolutionisto adopt a target-transfer approach, where audio out-A substantial body of prior work has explored put targets are transferred from vocalized record-the use of facial EMG for silent speech-to-text inings to silent recordings of the same utterances.terfaces (Jou et al., 2006; Schultz and Wand, 2010; We align the EMG features of the instance pairsKapur et al., 2018; Meltzner et al., 2018). Sevwith dynamic time warping (Rabiner and Juang,eral initial attempts have also been made to convert 1993), then make reﬁnements to the alignments usEMG signals to speech, similar to the task we apingcanonicalcorrelationanalysis(Hotelling,1936)proach in this paper (Toth et al., 2009; Janke and and audio feature outputs from a partially trained Diener, 2017; Diener et al., 2018). However, these model. The alignments can then be used to asso-works have focused on the artiﬁcial task of recovciate speech outputs with the silent EMG signals ering audio from EMG that was recorded during E , and these speech outputs are used as targetsvocalized speech, rather than the end-goal task of S for training a recurrent neural transduction model. generating from silent speech. In terms of signals in Figure 2, prior work learned a model for pro- We validate our method using both human and A E ducing audio from vocalized EMG and automatic metrics, and ﬁnd that a model trainedV V tested primarily on other vocalized EMG signals. with our target transfer approach greatly outperWhile one might hope that a model trained in this formsamodeltrainedonvocalizedEMGalone. On E way could directly transfer to silent EMG , Toth a closed-vocabulary domain (date and time expres-S et al. (2009) show that such a transfer causes a sub- sions §2.1), transcription word error rate (WER) stantial degradation in quality, which we conﬁrm from a human evaluation improves from 64% to in Section 4. This direct transfer from vocalized just 4%. On a more challenging open vocabulary models fails to account for differences between fea- domain (reading from books §2.2) intelligibility tures of the two speaking modes, such as a lack measurements improve by 20% – from 88% to of voicing in the vocal folds and other changes in 68% with automatic transcription or 95% to 75% articulation to suppress sound. with human transcription.

dataset contains nearly 20 hours of facial EMG signals from a single speaker. To our knowledge, the largest public EMG-speech dataset previously available contains just 2 hours of data (Wand et al., 2014), and many papers continue to use private datasets. We hope that this public release will encourage development on the task and allow for fair comparisons between methods.

2 Data Collection

We collect a dataset of EMG signals and timealigned audio from a single speaker during both silent and vocalized speech. Figure 2 shows an examplefromthedatacollected. Theprimaryportion of the dataset consists of parallel silent / vocalized data, where the same utterances are recorded using both speaking modes. These examples can be (E ,E ,A ) viewed as tuples of silent EMG, vo-S V V E calized EMG, and vocalized audio, where andV A are time-aligned. Both speaking modes of anV utterance were collected within a single session to ensure that electrode placement is consistent between them. For some utterances, we record only the vocalized speaking mode. We refer to these instances as non-parallel data, and represent them (E ,A ). with the tuple Examples are segmentedV V at the utterance level. The text that was read is included with each instance in the dataset, and is used as a reference when evaluating intelligibility in Section 4. For comparison, we record data from two domains: a closed vocabulary and open vocabulary condition, which are described in Sections 2.1 and 2.2 below. Section 2.3 then provides additional details about the recording setup.

2.1 Closed Vocabulary Condition Like other speech-related signals, the captured EMG signals from a particular phoneme may look different depending on its context. For this reason, our initial experiments will use a more focused vocabularysetbeforeexpandingtoalargevocabulary in Section 2.2 below. To create a closed-vocabulary data condition, we generate a set of date and time expressions for reading. These expressions come from a small <month> set of templates such as “<weekday> <year>” which are ﬁlled in with randomly selected values (over 50,000 unique utterances are

Closed Vocabulary Condition Parallel silent / vocalized speech (E ,E ,A ) S V V 26 minutes silent / 30 minutes vocalized Single session 500 utterances Average of 4 words per utterance 67 words in vocabulary

Table 1: Closed vocabulary data summary

thepropertiesofthedatacollectedinthiscondition. A validation set of 30 utterances and a test set of 100 utterances are selected randomly, leaving 370 utterances for training.

2.2 Open Vocabulary Condition The majority of our data was collected with openvocabulary sentences from books. We use public 1domainbooksfromProjectGutenberg. Unlikethe closed-vocabulary data which is collected in a single sitting, the open-vocabulary data is broken into multiple sessions where electrodes are reattached before each session and may have minor changes in position between different sessions. In addition to sessions with parallel silent and vocalized utterances, we also collect non-parallel sessions with only vocalized utterances. A summary of dataset features is shown in Table 2. We select a validation and test set randomly from the silent parallel EMG data, with 30 and 100 utterances respectively. Note that during testing, we use only the silent EMG E recordings , so the vocalized recordings of theS test utterances are unused.

2.3 Recording Details EMG signals are recorded using an OpenBCI Cyton Biosensing Board and transmitted to a computer over WiFi. Eight channels are collected at a sample rate of 1000 Hz. The electrode locations are described in Table 3. Gold-plated electrodes are used with Ten20 conductive electrode paste. We use a monopolar electrode conﬁguration, with a shared reference electrode behind one ear. An electrode connected to the Cyton board’s bias pin is placed behind the other ear to actively cancel common-mode interference. A high pass Butterworth ﬁlter with cutoff 2 Hz is used to remove offset and drift in the collected signals, and AC

Parallel Silent / Vocalized Speech (E ,E ,A ) S V V 3.6 hours silent / 3.9 hours vocalized Average session has 30 min. of each mode 1588 utterances Non-parallel Vocalized Speech (E ,A )V V 11.2 hours Average session length 67 minutes 5477 utterances Total 18.6 hours Average of 16 words per utterance 9828 words in vocabulary

Table 2: Open vocabulary data summary

Location 1 left cheek just above mouth 2 left corner of chin 3 below chin back 3 cm 4 throat 3 cm left from Adam’s apple 5 mid-jaw right 6 right cheek just below mouth 7 right cheek 2 cm from nose 8 back of right cheek, 4 cm in front of ear ref below left ear bias below right ear

Table 3: Electrode locations.

electrical noise is removed with notch ﬁlters at 60 Hz and its harmonics. Forward-backward ﬁlters are used to avoid phase delay. Audio is recorded from a built-in laptop microphone at 16kHz. Background noise is reduced using a spectral gating algorithm,2 and volume is normalized across sessions based on peak root-meansquare levels.

3 Method

Our method is built around a recurrent neural transduction model from EMG features to time-aligned speech features (Section 3.1). We will denote the featurized version of the signals used by the transE(cid:48) A(cid:48) duction model and for EMG and auV S/V dio respectively. When training solely on vocal-

straightforward. However, our experiments show that training on vocalized EMG alone leads to poor performance when testing on silent EMG (Section 4) because of differences between the two speaking modes. A core contribution of our work is a method of training the transducer model on silent EMG signals, which no longer have time-aligned audio to use as training targets. We brieﬂy describe our method here, then refer to section Section 3.2 for more details. Using a set of utterances recorded in both silent and vocalized speaking modes, we ﬁnd alignments between the two recordings and use them to associate speech features from the vocalA(cid:48) E(cid:48)ized instance ( ) with the silent EMG . The V S alignment is initially found using dynamic time warping between EMG signals and then is reﬁned using canonical correlation analysis (CCA) and predicted audio from a partially trained model. Finally, to generate audio from predicted speech features, we use a WaveNet decoder, as described in Section 3.3.

3.1 EMG to Speech Feature Transducer When converting EMG input signals to audio outputs, ourﬁrststepistouseabidirectionalLSTMto convert between featurized versions of the signals, E(cid:48) A(cid:48). and Both feature representations operate at the same frequency, 100 Hz, so that each EMG E(cid:48)[i] input corresponds to a single time-aligned (cid:48)output A [i]. Our primary features for representing EMG signals are the time domain features from Jou et al. (2006), which are commonly used in the EMG-speech-to-text literature. After splitting the signal from each channel into low and highx frequency components (x and ) using a low high triangular ﬁlter with cutoff 134 Hz, the signal is windowedwithaframelengthof27msandshiftof 10 ms. For each frame, ﬁve features are computed as follows: (cid:34) 1 1 1(cid:88) (cid:88) (cid:88) 2 2(x [i]) , x [i], (x [i]) , low low highn n n i i i (cid:35) 1 (cid:88) |x [i]|, )high ZCR(x high n i where ZCR is the zero-crossing rate. In addition to the time domain features, we also append magnitude values from a 16-point Short-time Fourier

sult in a total of 112 features to represent the 8 EMG channels. Speech is represented with 26 Melfrequency cepstral coefﬁcients (MFCCs) from 27 ms frames with 10 ms stride. All EMG and audio features are normalized to approximately zero mean and unit variance before processing. To help the model to deal with minor differences in electrode placement across sessions, we represent each session with a 32 dimensional session embedding and append the session embedding to the EMG features across all timesteps of an example before feeding into the LSTM. The LSTM model itself consists of 3 bidirectional LSTM layers with 1024 hidden units, followed by a linear projection to the speech feature 0.5dimension. Dropout is used between all layers, as well as before the ﬁrst LSTM and after the last LSTM. The model is trained with a mean squared error loss against time-aligned speech features using the Adam optimizer. The initial learning rate .001, is set to and is decayed by half after every 5 epochs with no improvement in validation loss. We evaluate a loss on the validation set at the end of every epoch, and select the parameters from the epoch with the best validation loss as the ﬁnal model.

3.2 Audio Target Transfer

To train the EMG to speech feature transducer, we need speech features that are time-aligned with the EMG features to use as target outputs. However, when training with EMG from silent speech, simultaneously-collected audio recordings do not have any audible speech to use as targets. In this section, we describe how parallel utterances, as described in Section 2, can be used to transfer audio feature labels from a vocalized recording to a silent one. More concretely, given a tuple (E(cid:48) ,E(cid:48) ,A(cid:48) ) of features from silent speech EMG, S V V vocalized speech EMG, and vocalized speech auE A dio, where and are collected simultaneV V ˜(cid:48)ously, A we estimate a set of audio features that S E(cid:48) time-align with and represent the output that S we would like our transduction network to predict. A diagram of the method can be found in Figure 3. Our alignment will make use of dynamic time warping (DTW) (Rabiner and Juang, 1993), a dynamic programming algorithm for ﬁnding a minimum-cost monotonic alignment between two

Figure 3: Our audio target transfer method for training E on silent EMG . Details in Section 3.2.S

i the minimum cost of alignment between the ﬁrst itemsins andtheﬁrstj itemsins . Therecursive1 2 d[i,j] = δ[i,j] + step used to ﬁll this table is min(d[i−1,j],d[i,j −1],d[i−1,j −1]), δ[i,j] s [i] where is the local cost of aligning with 1 s [j]. After the dynamic program, we can follow2 backpointers through the table to ﬁnd a path of (i,j) pairs representing an alignment. Although i the path is monotonic, a single position may j. repeat several times with increasing values of We take the ﬁrst pair from any such sequence to a is [i] → j form a mapping from every position s 1 2 s j s in to a position in . 1 2 For our audio target transfer, we perform DTW (cid:48) (cid:48)as s = E s = E described above with and . 1 2 S V Initially, we use euclidean distance between the (cid:48) (cid:48)features E E of and for the alignment cost S V [i,j] = (cid:107)E(cid:48) [i]−E(cid:48) [j](cid:107)), (δ but will describeEMG S V several reﬁnements to this choice in Sections 3.2.1 and 3.2.2 below. DTW results in an alignment (cid:48)a [i] → j j E that tells us a position in for evSV V i E(cid:48) ery position in . We can then create a warped S ˜(cid:48) (cid:48)audio A E feature sequence that aligns with usS S˜ A(cid:48) [i] = A(cid:48) [a [i]]. ing During training of the SV S V ˜(cid:48)EMG A to audio transduction model, we will use S ˆ(cid:48)as our targets for the transduction outputsA when S calculating a loss. This procedure of aligning signals to translate between them is reminiscent of some DTW-based methods for the related task of voice conversion (Kobayashi and Toda, 2018; Desai et al., 2009). The difference between these tasks is that our task ,E ,A )andmustaccount operatesontriples(E S V V forthedifferenceinmodalitybetweentheinputES A and output , while voice conversion operates V in a single modality with examples of the form

embeddings to allow the model to specialize to each speaking mode. Each training batch contains samples from both modes mixed together. For the open vocabulary setting, the full set of examples to ˜(cid:48)sample (cid:48) (E ,A ) from has 3 sources: created from S S (E ,A ) parallel utterances, from the vocalized V V (E ,A )V recording of the parallel utterances, and V from the non-parallel vocalized recordings.

3.2.1 CCA (cid:48) (cid:48)While E E directly aligning EMG features and S V can give us a rough alignment between the signals, doing so ignores the differences between the two signals that lead us to want to train on the silent signals in the ﬁrst place (e.g. inactivation of the vocal folds and changes in manner of articulation to prevent frication). To better capture correspondences between the signals, we use canonical correlation analysis (CCA) (Hotelling, 1936) to ﬁnd components of the two signals which are more highly correlated. Given a number of paired vec(v ,v ), P tors CCA ﬁnds linear projections and 1 2 1 P that maximize correlation between correspond-2 P v P v ing dimensions of and . 1 1 2 2 To get the initial pairings required by CCA, we use alignments found by DTW with the raw EMG (cid:48)feature δ E distance . We aggregate aligned EMG S E(cid:48) and features over the entire dataset and feed V thesetoa CCAalgorithmtoget projectionsP andS P . CCA allows us to choose the dimensionality V of the space we are projecting to, and we use 15 dimensions for all experiments. Using the projections from CCA, we deﬁne a new cost for DTW (cid:13) (cid:13)(cid:48) (cid:48) δ [i,j] = E [i]−P E [j](cid:13)CCA (cid:13)P S V S V Our use of CCA for DTW is similar to Zhou and Torre (2009), which combined the two methods for useinaligninghumanposedata,butwefoundtheir iterative approach did not improve performance compared to a single application of CCA in our setting.

3.2.2 Reﬁnement with Predicted Audio So far, our alignments between the silent and vocalized recordings have relied solely on distances between EMG features. In this section, we propose an additional alignment distance term that

an EMG-based distance, ourVnew cost for DTW becomes (cid:13) (cid:13) ˆ(cid:48) (cid:48) (cid:13) (cid:13)δ [i,j] = δ [i,j]+λ A [i]−A [j] full CCA (cid:13) (cid:13) S V

λ where is a hyperparameter to control the relative λ = 10 weight of the two terms. We use for all experiments in this paper. When training a transducer model using predicted-audio alignment, we perform the ﬁrst fourepochsusingonlyEMG-basedalignmentcosts δ . Then, at the beginning of the ﬁfth epoch, we CCA use the partially-trained transducer model to comδ pute alignments with cost . From then on, we full re-compute alignments every ﬁve epochs of training.

3.3 WaveNet Synthesis To synthesize audio from speech features, we use a WaveNet decoder (van den Oord et al., 2016), which generates the audio sample by sample con(cid:48)ditioned A on MFCC speech features . WaveNet is capable of generating fairly natural sounding speech, in contrast to the vocoder-based synthesizer used in previous EMG-to-speech papers, whichcausedsigniﬁcantdegradationinnaturalness (Janke and Diener, 2017). Our full synthesis model consists of a bidirectional LSTM of 512 dimensions, a linear projection down to 128 dimensions, and ﬁnally the WaveNet decoder which generates samplesat 16kHz. We usea WaveNet implementa3tion from NVIDIA which provides efﬁcient GPU inference. WaveNet hyperparameters can be found in Appendix A. During training, the model is given gold speech features as input, which we found to work better than training from EMG-predicted features. Due to memory constraints we do not use any batching during training, but other optimization hyperparameters are the same as those from Section 3.1.

4 Experiments

In this section, we run experiments to measure intelligibility of audio generated by our model from E silent EMG signals . Since prior work has S E trained only on vocalized EMG signals , weV

identical architecture to those used by our method, but are not trained with silent EMG using our target transfer approach. Since one may hypothesize that most of the differences between silent and vocalized EMG will take place near the vocal folds, we also test a variant of this baseline where the electrode placed on the neck is ignored. We ﬁrst test on the closed vocabulary data described in Section 2.1, then on the open vocabulary data from Section 2.2. On the open vocabulary data, we also run ablations to evaluate different alignment reﬁnements with CCA and predicted audio (see Sections 3.2.1 and 3.2.2).

4.1 Closed Vocabulary Condition We begin by testing intelligibility on the closed vocabulary date and time data with a human transcription evaluation. The human evaluator is given a set of 20 audio output ﬁles from each model being tested (listed below) and is asked to write out in words what they heard. The ﬁles to transcribe are randomly shufﬂed, and the evaluator is not told that the outputs come from different systems. They are told that the examples will contain dates and times, but are not given any further information about what types of expressions may occur. The full text of the instructions provided to the evaluator can be found in Appendix B. We compare the transcriptions from the human evaluator to the original text prompts that were read during data collection to compute a transcription word error rate (WER):

substitutions+insertions+deletions = WER reference length

Lower WER values indicate better models. Using this evaluation, we compare three different models: a direct transfer baseline trained only on vocalized EMG signals, a variant of this baselinewherethethroatelectrodeisremovedtoreduce divergence between speaking modes, and our full model trained on silent EMG using target-transfer. All three models were trained on open vocabulary

Our model 3.6

Table4: Resultsofahumanintelligibilityevaluationon the closed vocabulary data. Lower WER is better. Our model greatly outperforms both variants of the direct transfer baseline.

data (Section 2.2) before being ﬁne-tuned on the closed vocabulary training set. A single WaveNet model is used to synthesize audio for all three models and was also trained on the open vocabulary data before being ﬁne-tuned in-domain. The results of our evaluation are shown in Table 4. We ﬁrst observe that removing the throat electrode substantially improves intelligibility for the direct transfer baseline. Although this modiﬁcation removes potentially useful information, it also removes divergence between the silent and vocalized EMG signals. Its relative success further motivates the need for methods to account for the differences in the two modes, such as our targettransfer approach. However, even with the throatremoval modiﬁcation, the direct transfer approach is still only partially intelligible. A model trained with our full approach, including CCA and predicted-audio alignment, achieves a WER of 3.6%. This result represents a high level of intelligibility and a 94% relative error reduction from the strongest baseline.

4.2 Open Vocabulary Condition Similar to our evaluation in Section 4.1, we use a transcription WER to evaluate intelligibility of model outputs in the open vocabulary condition. For the open vocabulary setting, we evaluate both with a human transcription and with transcriptions from an automatic speech recognizer.

4.2.1 Human Evaluation Our human evaluation with open vocabulary outputs follows the same setup as the closed vocabulary evaluation. Transcripts are collected for 20 audio outputs from each system, with a random

Direct transfer baseline 91.2 Without throat electrode 88.0 68.0 Our model Without CCA 69.8 Without audio alignment 76.5

Table 5: Results of an automatic intelligibility evaluation on open vocabulary data. Lower WER is better.

The results of this evaluation are a 95.1% WER for the direct transfer baseline and 74.8% WER for our system. While the intelligibility is much lower thanintheclosedvocabularycondition,ourmethod still strongly out-performs the baseline with a 20% absolute improvement.

4.2.2 Automatic Evaluation

In addition to the human evaluation, we also perform an automatic evaluation by transcribing system outputs with a large-vocabulary automatic speech recognition (ASR) system. Using an automatic transcription allows for much faster and more reproducible comparisons between methods comparedtoahumanevaluation. Forourautomatic speech recognizer, we use the open source imple5mentation of DeepSpeech from Mozilla (Hannun et al., 2014). Running the recognizer on the original vocalized audio recordings from the test set resultsinaWERof9.5%,whichrepresentsalower bound for this evaluation. Our automatic evaluation results are shown in Table 5. While the absolute WER values for the ASR evaluation do not perfectly match those of the human transcriptions, both evaluations show a 20% improvement of our system over the best baseline. Given this correlation between evaluations and the many advantages of automated evaluation, we will use the automatic metric throughout the rest of this work and recommend its use for comparisons in future work. We also run ablations of the two alignment reﬁnement methods from Sections 3.2.1 and 3.2.2 and include results in Table 5. We see that both reﬁnements have a positive effect on performance, though the impact of aligning with predicted audio is greater.

80 R E W 70

60 0 5 10 15 20 Data Amount (Hours)

Figure 4: Effect of data amount on intelligibility.

4.3 Additional Experiments Inthefollowingsubsections,weperformadditional experimentsontheopenvocabularydatatoexplore the effect of data size and choice of electrode positions. These experiments are all evaluated using the automatic transcription method described in Section 4.2.

4.3.1 Data Size In this section we explore the effect of dataset size on model performance. We train the EMG-tospeech transducer model on various-sized fractions of the dataset, from 10% to 100%, and plot the resulting WER. We select from the parallel (silent and vocalized) and non-parallel (vocalized only) portions proportionally here, but will re-visit the difference later. Although data size also affects WaveNet quality, we use a single WaveNet trained on the full dataset for all evaluations to focus on EMG-speciﬁc data needs. Figure 4 shows the resulting intelligibility measurementsforeachdatasize. Aswouldbeexpected, the rate of improvement is larger when data sizes are small. However, there does not seem to be a plateau in performance, as improvements continueevenwhenincreasingdatasizebeyondﬁfteen hours. These continued gains suggest that collecting additional data could provide more improvement in the future. We also train a model without the non-parallel vocalized data (vocalized recordings with no associated silent recording; see Section 2). A model trained without this data has a WER of 71.6%, a loss of 3.6 absolute percentage points. This conﬁrms that non-parallel vocalized data can be useful for silent speech even though it contains only data from the vocalized speaking mode. However, if we compare this accuracy to a model where the same

the two data types (parallel and non-parallel), we see that removing a mixture of both types leads to a much larger performance decrease to 76% WER. This indicates that the non-parallel data is less important to the performance of our model, and suggeststhatfuturedatacollectioneffortsshouldfocus on collecting parallel utterances of silent and vocalized speech rather than non-parallel utterances of vocalized speech.

4.3.2 Removing Electrodes In this section, we experiment with models that operate on a reduced set of electrodes to assess the impact on performance and gain information about which electrodes are most important. We perform a random search to try to ﬁnd a subset of four electrodes that works well. More speciﬁcally, we sample 10 random combinations of four electrodes to remove (out of 70 possible combinations) and train a model with each. We then use validation loss to select the best models. The three best-performing models removed the following sets of electrodes (using electrode num  bering from Table 3): 1) 2) and3). Wenotethatelectrodes5,7,and 8 (which correspond with electrodes on the midjaw, upper cheek, and back cheek) appear in all of these, indicating that they may be contributing less to the performance of the model. However, the best model we tested with four electrodes did have substantially worse intelligibility compared to an eight-electrodemodel, with76.8%WERcompared to 68.0%. A model that removed only electrodes 5, 7, and 8 also performed substantially worse, with a WER of 75.3%.

5 Conclusion

Our results show that digital voicing of silent speech, while still challenging in open domain settings, shows promise as an achievable technology. We show that it is important to account for differences in EMG signals between silent and vocalized speaking modes and demonstrate an effective method of doing so. On silent EMG recordings from closed vocabulary data our speech outputs achieve high intelligibility, with a 3.6% transcription word error rate and relative error reduction of 95% from our baseline. We also signiﬁcantly improve intelligibility in an open vocabulary condition, with a relative error reduction over 20%. We hope that our public release of data will encourage

6others to further improve models for this task.

Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. 1618460.

References

Srinivas Desai, E Veera Raghavendra, B Yegnanarayana, Alan W Black, and Kishore Prahallad. 2009. Voice conversion using artiﬁcial neural net2009 IEEE International Conference works. In on Acoustics, Speech and Signal Processing, pages 3893–3896. IEEE.

L. Diener, G. Felsch, M. Angrick, and T. Schultz. 2018. Session-independent array-based EMG-tospeech conversion using convolutional neural netSpeech Communication; 13th ITG- works. In Symposium, pages 1–5.

Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, SanjeevSatheesh,ShubhoSengupta,AdamCoates,etal. 2014. Deep speech: Scaling up end-to-end speech arXiv preprint arXiv:1412.5567. recognition.

Harold Hotelling. 1936. Relations between two sets of variates.

M. Janke and L. Diener. 2017. EMG-to-speech: Direct generation of speech from facial electromyographic IEEE/ACM Transactions on Audio, Speech, signals. and Language Processing, 25(12):2375–2385.

Szu-Chen Stan Jou, Tanja Schultz, Matthias Walliczek, Florian Kraft, and Alexander H. Waibel. 2006. Towards continuous speech recognition using surface INTERSPEECH. electromyography. In

Arnav Kapur, Shreyas Kapur, and Pattie Maes. 2018. Alterego: A personalized wearable silent speech in23rd International Conference on Intelli- terface. In gent User Interfaces, pages 43–53.

KazuhiroKobayashiandTomokiToda.2018. sprocket: Odyssey, Open-source voice conversion software. In pages 203–210.

Geoffrey S Meltzner, James T Heaton, Yunbin Deng, Gianluca De Luca, Serge H Roy, and Joshua C Kline. 2017. Silent speech recognition as an alternative communication device for persons with larynIEEE/ACM transactions on audio, speech, gectomy. and language processing, 25(12):2386–2398.

6Our dataset can be downloaded from https://doi.org/10.5281/zenodo.4064408 and code is available at https://github.com/dgaddy/silent_speech.

Geoffrey S Meltzner, James T Heaton, Yunbin Deng, GianlucaDeLuca,SergeHRoy,andJoshuaCKline. 2018. Development of sEMG sensors and algoJournal of neu- rithms for silent speech recognition. ral engineering, 15(4):046031.

Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and Koray Kavukcuoglu. 2016. WaveNet: A generative model ArXiv, for raw audio. abs/1609.03499.

Fun- LawrenceRabinerandBiing-HwangJuang.1993. damentals of speech recognition. Prentice Hall.

Tanja Schultz and Michael Wand. 2010. Modeling coarticulation in EMG-based continuous speech Speech Communicationrecognition. , 52(4):341– 353.

Arthur R. Toth, Michael Wand, and Tanja Schultz. 2009. Synthesizing speech from electromyography INTER- using voice transformation techniques. In SPEECH.

Michael Wand, Matthias Janke, and Tanja Schultz. 2014. The EMG-UKA corpus for electromyoINTERSPEECH. graphic speech processing. In

Feng Zhou and Fernando Torre. 2009. Canonical time Ad-warping for alignment of human behavior. In vances in neural information processing systems, pages 2286–2294.

A WaveNet Hyperparameters

Hyperparameter Value n_in_channels 256 n_layers 16 max_dilation 128 n_residual_channels 64 n_skip_channels 256 n_out_channels 256 n_cond_channels 128 upsamp_window 432 upsamp_stride 160

B Human Evaluator Instructions

The instructions given to the human evaluator are as follows: “Please listen to each of the attached sound ﬁles and write down what you hear as best you can. There are 60 ﬁles, each of which will contain an expression of some date or time. Write yourtranscriptionsintoaspreadsheetsuchasExcel

on Thursday7 Many of the clips may be difﬁcult to hear. If this is the case, write whatever words you are able to make out, even if it does not five form a complete expression. For example: two pm on If you cannot make out any words, leave the corresponding row blank.”

C Additional Data Collection Details

During data collection, text prompts consisting of a single sentence to be read are displayed on a screen. After reading the sentence, the subject pressed a key to advance to the next sentence. If they were unhappy with a recording, they could press another key to re-record an utterance. A realtime display of EMG signals was used to monitor the electrodes for excessive noise. During silent speech, the subject was instructed to mouth words as naturally as possible without producing sound.

D Additional Reproducibility Information

Models were trained for up to two days on a single K80 GPU. Hyperparameter search consisted of a mixture of manual and random search, typicallywithlessthan10runs. Hyperparameterswere chosen primarily based validation loss, with major designdecisionsalsobeingcheckedwithautomatic transcription evaluation.