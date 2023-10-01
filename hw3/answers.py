r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 128
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.3
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.01
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE 1.\n Limor. The COUNT'S palace"
    temperature = 0.6
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Dividing the corpus into sequences rather than training on the entirety of the text helps prevent overfitting.
This is because using the full text may cause the model to memorize the text instead of learning to generalize.
By dividing the corpus into sequences, the model is presented with various character sequences in different orders,
which helps mitigate this risk.
"""

part1_q2 = r"""
**Your answer:**
The hidden state’s length is independent of the sequence length, so when generating text,
it can be of a different length than the sequence.
"""

part1_q3 = r"""
**Your answer:**
During training, we carry over the hidden state between batches to aid in the model’s learning by preserving
relevant information in the hidden state. To maintain the original text’s order and pass the most appropriate
hidden state across training batches, we do not shuffle the order of batches.
"""

part1_q4 = r"""
**Your answer:**
1. The temperature parameter controls the degree of variability in the distribution.
Lowering the temperature increases the chance of selecting characters that the model deems more likely by making
the distribution more skewed towards high probability events.

2. When the temperature is extremely high, the distribution approaches uniformity.
This is because all the values in the distribution approach 1 as the temperature increases.

3. When the temperature is very low, high-probability events are significantly emphasized.
This is because the distribution becomes skewed and ‘spiky’, giving more weight to high probability events.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 512
    hypers['z_dim'] = 8
    hypers['x_sigma2'] = 0.0005
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""**Your answer:** 
If x_sigma2 is low (approaching 0), it implies a strong belief that the reconstructions should be very close to the 
original inputs, in other words - a low x_sigma2 value makes the loss function more sensitive to differences between 
the input and its reconstruction, which could potentially lead to overfitting if the model learns to reproduce the 
training data too closely.

High value of x_sigma2 means that there is more uncertainty or noise in the reconstructed data.
The model will not penalize small differences between the original and reconstructed 
data as heavily, leading to a more robust but possibly less accurate model.
If x_sigma2 is set too high the model may underfit, meaning it won't learn the underlying structure of the data well
enough and the quality of the generated faces could be lower.
"""

part2_q2 = r"""
The purpose of the reconstruction loss is to measure the difference between the original input data and the data reconstructed by 
the VAE. By minimizing the reconstruction loss, the VAE learns to create data that is as close as possible to the 
original input data.

The KL Divergence loss measures how much the distribution of latent variables (the approximate 
posterior) deviates from a target distributio which in our case is a normal distribution. By minimizing this 
loss, the VAE ensures that the distribution of latent variables closely aligns with this target distribution.

key benefits:
Regularization:  By aligning the approximate posterior distribution with a prior standard normal distribution,
it ensures that the model does not learn overly complex representations that could lead to overfitting to the training data.
Data Generation: Because the distribution in the latent space is regular and known (normal),
we can sample points from this distribution, decode these points to generate new, varied data.
"""

part2_q3 = r"""In VAEs, the ultimate goal is to find the model that best represents our observed data. This is 
encapsulated by the concept of maximizing the evidence distribution $p(\mathbf{X})$. However, direct computation of 
this evidence is typically unfeasible due to the complexity introduced by latent variables.

To circumvent this issue, we maximize the ELBO, a *lower bound on the log evidence*, which is more computationally 
manageable. While the actual optimization process starts with maximizing the ELBO, the principle of maximizing the 
evidence guides the entire learning process.

By maximizing the ELBO, we're ensuring that our VAE is getting better at both reconstructing the original data and 
keeping the latent variables distributed in a way that we can manage. This allows the VAE to generate new data that 
resembles the original data it was trained on!

"""

part2_q4 = r"""
Modeling the log of the variance in the VAE encoder, rather than the variance itself, is primarily for 
numerical stability and ease of computation.
 
Why log?
Variance values are always non-negative, and using the log 
transformation ensures this constraint naturally. The log variance also *spans* the entire real line, which can make 
optimization easier as it avoids the potential problem of the optimization getting "stuck" at zero. Additionally, 
the log transformation can help to smooth out the optimization landscape and lessen the impact of extreme values.

Modeling log-variance in VAEs aligns with the assumption of a multivariate Gaussian with a diagonal covariance 
matrix. The diagonal entries of the covariance matrix represent variances of individual dimensions. By using 
log-variance, we can conveniently ensure these values remain positive, as required in a Gaussian. Upon 
exponentiating, we get the actual variance elements, simplifying computations and maintaining numerical stability."""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=0,
        num_heads=0,
        num_layers=0,
        hidden_dim=0,
        window_size=0,
        droupout=0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
The DistilBERT model on the IMDB dataset for sentiment analysis, the approach of retraining all the 
parameters in the model yielded better results compared to freezing all weights except for the last two linear 
layers. several reasons that could explain this result:

<b>Task Specificity:</b> Sentiment analysis, especially on something as context-specific as movie reviews, might involve 
understanding nuances that were not fully captured in the initial pretraining phase of the model. By fine-tuning all 
layers, we probably allow the model to better adapt to these nuances and thus achieve better performance. 

<b>Data Volume:</b> The IMDB dataset large enough to produce better acc on this task.
it's more feasible to fine-tune all parameters of the model without worrying too much about overfitting.
Retraining the entire model allows it to better fit the specific distribution of IMDB data.

<b>Complexity of Task:</b> Sentiment analysis, particularly in the context of movie reviews, can involve complex language 
understanding beyond simple lexical cues. This could include understanding sarcasm, domain-specific lingo.
Such understanding may require tuning of more than just the last few layers of the model."""

part3_q2 = r"""
The multi-headed attention is vital for the model to understand the dependencies between different 
words in a sentence, it allows the model to capture the **context** of words in a sentence. 
Unfreezing and allowing them to be tuned could potentially improve the model's performance by enabling it to better capture 
task-specific dependencies in the data - such as sentiment analysis on movie reviews.

However(!!!)- there is one major potential risk..  it's possible that these layers already capture useful,
general-purpose representations from pre-training, and fine-tuning them might cause the model to 'forget' this information.

The model could still fine-tune to the task, but the results may not be as good as fine-tuning all layers or just the 
two linear ones. Here's why:

For sentiment analysis on the IMDB movie reviews dataset, which is a task that relies heavily on understanding the 
context and nuances in the reviews, freezing the multi-headed attention blocks could lead to inferior performance 
compared to the strategy of freezing only the last two linear layers. The multi-headed attention blocks play a key 
role in understanding the semantic meaning of words in their specific context, which is crucial for accurately 
classifying the sentiment of the reviews.
"""

part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""

# ==============
