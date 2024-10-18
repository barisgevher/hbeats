package com.example.hbeats

import android.annotation.SuppressLint
import android.os.Bundle
import android.os.PersistableBundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.Column
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.graphics.Color
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.HeartRateRecord
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import androidx.lifecycle.lifecycleScope
import com.example.hbeats.databinding.ActivityMainBinding
import com.google.android.gms.auth.api.identity.BeginSignInRequest
import com.google.android.gms.auth.api.identity.Identity
import com.google.android.gms.auth.api.identity.SignInClient
import com.google.android.gms.common.api.ApiException
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.auth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.launch
import java.text.DecimalFormat
import java.time.Instant
import java.time.ZonedDateTime
import java.time.temporal.ChronoUnit

private const val TAG = "Hbeats"

class MainActivity : ComponentActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var healthConnectClient: HealthConnectClient
    private val googleIdOption: GetGoogleIdOption = GetGoogleIdOption.Builder()
        .setFilterByAuthorizedAccounts(true)
        .setServerClientId(R.string.clientID.toString())
        .build()
//
//    private val request: GetCredentialRequest = GetCredentialRequest.Builder()
//        .addCredentialOption(googleIdOption)
//        .build()

    private val REQ_ONE_TAP = 2  // Can be any integer unique to the Activity
    private var showOneTapUI = true

    private lateinit var signInRequest: BeginSignInRequest

    val permissions = setOf(
        HealthPermission.getReadPermission(HeartRateRecord::class),
    )

    val requestPermissionActivityContract =
        PermissionController.createRequestPermissionResultContract()

    val requestPermissions =
        registerForActivityResult(requestPermissionActivityContract) { granted ->
            if (granted.containsAll(permissions)) {
                // Permissions successfully granted
            } else {
                // Lack of required permissions
            }
        }

    private lateinit var oneTapClient: SignInClient
    private lateinit var auth: FirebaseAuth

    private val activtiyResultLauncher =
        registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { result ->
            println("result -> $result")

            when (result.resultCode) {
                RESULT_OK -> {
                    try {
                        val credential = oneTapClient.getSignInCredentialFromIntent(result.data)
                        val idToken = credential.googleIdToken
                        println("id token: $idToken")

                        when {
                            idToken != null -> {
                                // Got an ID token from Google. Use it to authenticate
                                // with Firebase.
                                Log.d(TAG, "Got ID token.")

                                val firebaseCredential = GoogleAuthProvider.getCredential(idToken, null)
                                auth.signInWithCredential(firebaseCredential)
                                    .addOnCompleteListener { task ->
                                        println("addOnComplete")
                                        if(task.isSuccessful) {
                                            val user = auth.currentUser
                                            println("addOnComplete isSuccessful ${user?.displayName}")

                                            Log.i("UserFirebase", "${user?.displayName}")
                                        } else {
                                            Log.i("UserFirebase", "error ${task.exception}")

                                        }

                                }.addOnFailureListener {
                                    println("hata geldi gardeş: $it")
                                    }.addOnCanceledListener {
                                        println("hata")
                                    }
                            }

                            else -> {
                                // Shouldn't happen.
                                Log.d(TAG, "No ID token!")
                            }
                        }
                    } catch (e: ApiException) {
                        println("hata geldi gardeş: api exc $e")

                    }catch (e: Exception) {
                        println("hata geldi gardeş: e exc $e")

                    }
                }

                else -> {}
            }

        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i("log activity", "activity onCreated called")

//        healthConnectManager = HealthConnectManager(this)

        auth = Firebase.auth

        healthConnectClient = HealthConnectClient.getOrCreate(this)

        oneTapClient = Identity.getSignInClient(this)

        signInRequest = BeginSignInRequest.builder().setGoogleIdTokenRequestOptions(
            BeginSignInRequest.GoogleIdTokenRequestOptions.builder()
                .setSupported(true)
                .setServerClientId(getString(R.string.clientID))
                .setFilterByAuthorizedAccounts(false)
                .build()
        ).build()
//        requestPermissions = registerForActivityResult(healthConnectManager.requestPermissionActivityContract) { granted ->
//            lifecycleScope.launch {
//                if (granted.isNotEmpty() && healthConnectManager.hasAllPermissions()) {
//                    Toast.makeText(
//                        this@MainActivity,
//                        R.string.permission_granted,
//                        Toast.LENGTH_SHORT,
//                    ).show()
//                } else {
//                    AlertDialog.Builder(this@MainActivity)
//                        .setMessage(R.string.permission_denied)
//                        .setPositiveButton(R.string.ok, null)
//                        .show()
//                }
//            }
//        }

        setContent {
            val scope = rememberCoroutineScope()

            Column {
                Text(text = "hello world", color = Color.White)

                Button(onClick = {
                    if (checkAvailability()) {
//                    checkPermissions(false)
//                    scope.launch {
//                        checkPermissionsAndRun()
//                    }

                        oneTapClient.beginSignIn(signInRequest).addOnCompleteListener { task ->
                            if (task.isSuccessful) {
                                val intentSender = task.result.pendingIntent.intentSender
                                val intentSenderRequest =
                                    IntentSenderRequest.Builder(intentSender).build()
                                activtiyResultLauncher.launch(intentSenderRequest)
                            } else {
                                Log.i("error", "${task.exception}")
                            }

                        }
                    }
                }) {
                    Text(text = "tıkla ", color = Color.White)
                }
            }

        }
//
//        binding = ActivityMainBinding.inflate(layoutInflater)
//        setContentView(binding.root)


    }


    suspend fun hasAllPermissions(): Boolean {
        return healthConnectClient.permissionController.getGrantedPermissions()
            .containsAll(permissions)
    }


    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.connect) {
            if (checkAvailability()) {
                //checkPermissions(true)
            }
            return true
        }

        return super.onOptionsItemSelected(item)
    }


    suspend fun readHeartRate(start: Instant, end: Instant): Number {
        val request = ReadRecordsRequest(
            recordType = HeartRateRecord::class,
            timeRangeFilter = TimeRangeFilter.between(start, end),
        )
        val response = healthConnectClient.readRecords(request)
        if (response.records.isNotEmpty()) {
            val heartRateRecord = response.records.last()
            val sample = heartRateRecord.samples.lastOrNull()
            return sample?.beatsPerMinute ?: 0.0
        }
        return 0.0
    }


    private fun checkAvailability(): Boolean {
        when (HealthConnectClient.getSdkStatus(this)) {
            HealthConnectClient.SDK_UNAVAILABLE -> {
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        R.string.not_supported_description,
                        Toast.LENGTH_SHORT,
                    ).show()
                }
                return false
            }

            HealthConnectClient.SDK_UNAVAILABLE_PROVIDER_UPDATE_REQUIRED -> {
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        R.string.not_installed_description,
                        Toast.LENGTH_SHORT,
                    ).show()
                }
                return false
            }

            else -> {
                return true
            }
        }
    }

    suspend fun checkPermissionsAndRun() {
        val granted = healthConnectClient.permissionController.getGrantedPermissions()
        if (granted.containsAll(permissions)) {
            // Permissions already granted; proceed with inserting or reading data
            readAllData()
        } else {
            requestPermissions.launch(permissions)
        }
    }

    fun onRefresh(view: View) {
        if (checkAvailability()) {
            //checkPermissionsAndRun(true)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun readAllData() {
        lifecycleScope.launch {
            val startOfDay = ZonedDateTime.now().truncatedTo(ChronoUnit.DAYS)
            val now = Instant.now()


            val heartRate = readHeartRate(startOfDay.toInstant(), now)

            //binding.textHeartRate.text = heartRate.toString() // Display heart rate
            Log.i("Heart Rate data", "Data --> $heartRate ")

        }
    }
}
