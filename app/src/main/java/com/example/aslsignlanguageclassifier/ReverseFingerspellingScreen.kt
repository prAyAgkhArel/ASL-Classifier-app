@file:OptIn(
    androidx.compose.foundation.layout.ExperimentalLayoutApi::class,
    androidx.compose.material3.ExperimentalMaterial3Api::class
)

package com.example.aslsignlanguageclassifier

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp

@Composable
fun ReverseFingerspellingScreen() {
    var inputText by rememberSaveable { mutableStateOf("") }
    var mode by rememberSaveable { mutableStateOf(1) }

    val letters = inputText.uppercase().filter { it.isLetter() }.toList()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerHigh
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    "Reverse Fingerspelling",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    "Type a letter, word, or sentence to see ASL alphabet images.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(16.dp))

                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    SegmentedButton(
                        selected = mode == 0,
                        onClick = { mode = 0 },
                        shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3)
                    ) { Text("Letter") }

                    SegmentedButton(
                        selected = mode == 1,
                        onClick = { mode = 1 },
                        shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3)
                    ) { Text("Word") }

                    SegmentedButton(
                        selected = mode == 2,
                        onClick = { mode = 2 },
                        shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3)
                    ) { Text("Sentence") }
                }

                Spacer(modifier = Modifier.height(16.dp))

                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = mode != 2,
                    label = { Text("Type here") },
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Characters
                    )
                )
            }
        }

        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Sign Preview", style = MaterialTheme.typography.titleMedium)

                Spacer(modifier = Modifier.height(12.dp))

                if (letters.isEmpty()) {
                    Text(
                        "Type text to preview fingerspelling.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        letters.forEach { letter ->
                            LetterSignCard(letter = letter)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LetterSignCard(letter: Char) {
    val imageRes = aslLetterImageRes(letter)

    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (imageRes != null) {
                Image(
                    painter = painterResource(id = imageRes),
                    contentDescription = "Sign for $letter",
                    modifier = Modifier.size(84.dp),
                    contentScale = ContentScale.Fit
                )
            } else {
                Text(
                    text = letter.toString(),
                    style = MaterialTheme.typography.headlineMedium
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = letter.toString(),
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}

private fun aslLetterImageRes(letter: Char): Int? {
    return when (letter.uppercaseChar()) {
        'A' -> R.drawable.a
        'B' -> R.drawable.b
        'C' -> R.drawable.c
        'D' -> R.drawable.d
        'E' -> R.drawable.e
        'F' -> R.drawable.f
        'G' -> R.drawable.g
        'H' -> R.drawable.h
        'I' -> R.drawable.i
        'J' -> R.drawable.j
        'K' -> R.drawable.k
        'L' -> R.drawable.l
        'M' -> R.drawable.m
        'N' -> R.drawable.n
        'O' -> R.drawable.o
        'P' -> R.drawable.p
        'Q' -> R.drawable.q
        'R' -> R.drawable.r
        'S' -> R.drawable.s
        'T' -> R.drawable.t
        'U' -> R.drawable.u
        'V' -> R.drawable.v
        'W' -> R.drawable.w
        'X' -> R.drawable.x
        'Y' -> R.drawable.y
        'Z' -> R.drawable.z
        else -> null
    }
}