package com.example.hbeats


import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.GMobiledata
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.HeartRateRecord
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import androidx.lifecycle.lifecycleScope
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.rememberLottieComposition
import com.google.android.gms.auth.api.identity.BeginSignInRequest
import com.google.android.gms.auth.api.identity.Identity
import com.google.android.gms.auth.api.identity.SignInClient
import com.google.android.gms.common.api.ApiException
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.ktx.auth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.toObject
import com.google.firebase.ktx.Firebase
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.android.Android
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.ZonedDateTime
import java.time.temporal.ChronoUnit
import kotlin.random.Random


const val TAG = "Hbeats"

class MainActivity : ComponentActivity() {
    private lateinit var healthConnectClient: HealthConnectClient

    val permissions = setOf(
        HealthPermission.getReadPermission(HeartRateRecord::class),
    )

    val requestPermissionActivityContract =
        PermissionController.createRequestPermissionResultContract()

    val requestPermissions =
        registerForActivityResult(requestPermissionActivityContract) { granted ->
            if (granted.containsAll(permissions)) {
                Log.i("izinler verildi", "gerekli izinler verildi")
            } else {
                Log.i("izinler verilmedi", "gerekli izinler alınamadı")
            }
        }

//
//    private val googleIdOption: GetGoogleIdOption =
//        GetGoogleIdOption.Builder().setFilterByAuthorizedAccounts(true)
//            .setServerClientId(R.string.clientID.toString()).build()


    private lateinit var signInRequest: BeginSignInRequest
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

                                Log.d(TAG, "Got ID token.")

                                val firebaseCredential =
                                    GoogleAuthProvider.getCredential(idToken, null)
                                auth.signInWithCredential(firebaseCredential)
                                    .addOnCompleteListener { task ->
                                        println("addOnComplete")
                                        if (task.isSuccessful) {
                                            val user = auth.currentUser
                                            println("addOnComplete isSuccessful ${user?.displayName}")

                                            Log.i("UserFirebase", "${user?.displayName}")


                                        } else {
                                            Log.i("UserFirebase", "error ${task.exception}")

                                        }

                                    }.addOnFailureListener {
                                        println("hata yakalandı: $it")
                                    }.addOnCanceledListener {
                                        println("hata")
                                    }
                            }

                            else -> {

                                Log.d(TAG, "No ID token!")
                            }
                        }
                    } catch (e: ApiException) {
                        println("api exception oluştu: api exc $e")

                    } catch (e: Exception) {
                        println("hata oluştu : e exc $e")

                    }
                }
            }

        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i("log activity", "activity onCreated called")
        healthConnectClient = HealthConnectClient.getOrCreate(this)

//        lifecycleScope.launch {
//            checkPermissionsAndRun()
//        }

        setContent {
            Navigation()
        }
    }


    suspend fun hasAllPermissions(): Boolean {
        return healthConnectClient.permissionController.getGrantedPermissions()
            .containsAll(permissions)
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
//            readAllData()
        } else {
            requestPermissions.launch(permissions)
        }
    }

    @SuppressLint("SetTextI18n")
    private suspend fun readAllData() : Number {
        val startOfDay = ZonedDateTime.now().truncatedTo(ChronoUnit.DAYS)
        val now = Instant.now()

        val heartRate = readHeartRate(startOfDay.toInstant(), now)
        return heartRate
    }



    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    @Composable
    fun LoginScreen(navController: NavHostController) {
        val composition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.heart_smart_watch))

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(color = Color.White)
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Welcome to HBeats!",
                style = MaterialTheme.typography.headlineLarge,
                color = Color.Black
            )
            Spacer(modifier = Modifier.height(16.dp))
            LottieAnimation(
                composition,
                reverseOnRepeat = true,
                iterations = LottieConstants.IterateForever
            )

            Text(
                modifier = Modifier.fillMaxWidth(),
                text = stringResource(R.string.text_login_desc),
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = {
                    signIn(
                        onSuccess = {
                            navController.navigate(Route.home.name)
                        },
                        onError = { error ->
                            Log.i("error", error.message.toString())
                        }
                    )
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White
                ),
                border = BorderStroke(
                    width = 2.dp,
                    color = Color.DarkGray
                )
            ) {
                Icon(
                    imageVector = Icons.Default.GMobiledata,
                    contentDescription = null,
                    tint = Color.Blue
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = "Sign in with Google")
            }
        }
    }


    private fun signIn(
        onSuccess: () -> Unit,
        onError: (Exception) -> Unit
    ) {
        if (checkAvailability()) {
            var intentSenderRequest: IntentSenderRequest
            auth = Firebase.auth

            oneTapClient = Identity.getSignInClient(this)

            signInRequest = BeginSignInRequest.builder().setGoogleIdTokenRequestOptions(
                BeginSignInRequest.GoogleIdTokenRequestOptions.builder().setSupported(true)
                    .setServerClientId(getString(R.string.clientID))
                    .setFilterByAuthorizedAccounts(false).build()
            ).build()

            oneTapClient.beginSignIn(signInRequest).addOnCompleteListener { task ->
                if (task.isSuccessful) {

                    val intentSender = task.result.pendingIntent.intentSender
                    intentSenderRequest = IntentSenderRequest.Builder(intentSender).build()
                    activtiyResultLauncher.launch(intentSenderRequest)
                    Log.i("Sign in", "process is Succesful continue to next page")

                    onSuccess()
                } else {
                    Log.i("error", "${task.exception}")
                    onError(task.exception ?: Exception("Login error. Try Again!"))
                }


            }

        }
        val mAuth = FirebaseAuth.getInstance()
        val currentUser: FirebaseUser? = mAuth.currentUser
        if (currentUser != null) {
            Log.i("UserFirebase", "User is signed in")
            Log.i("UserFirebase", "${currentUser.displayName}")
        }

    }


    @Composable
    fun MainScreen(
        navController: NavHostController,
    ) {
        var risk by remember { mutableStateOf(false) }
        var showRiskDialog by remember { mutableStateOf(false) }

        val heartRateState = remember { mutableStateOf(0f as Number) }

        val coroutineScope = rememberCoroutineScope()
        var loading by remember { mutableStateOf(true) }

        var age by remember { mutableStateOf("") }
        var cholesterol by remember { mutableStateOf("") }
        var diabetes by remember { mutableStateOf("") }
        var familyHistory by remember { mutableStateOf("") }
        var smoking by remember { mutableStateOf("") }
        var obesity by remember { mutableStateOf("") }
        var alcoholConsumption by remember { mutableStateOf("") }
        var exerciseHoursPerWeek by remember { mutableStateOf("") }
        var diet by remember { mutableStateOf("") }
        var previousHeartProblems by remember { mutableStateOf("") }
        var medicationUse by remember { mutableStateOf("") }
        var stressLevel by remember { mutableStateOf("") }
        var sedentaryHoursPerDay by remember { mutableStateOf("") }
        var bmi by remember { mutableStateOf("") }
        var triglycerides by remember { mutableStateOf("") }
        var physicalActivityDaysPerWeek by remember { mutableStateOf("") }
        var sleepHoursPerDay by remember { mutableStateOf("") }
        var isMale by remember { mutableStateOf("") }
        var systolicPressure by remember { mutableStateOf("") }
        var diastolicPressure by remember { mutableStateOf("") }



        Scaffold(
            topBar = {
                TopAppBar(
                    backgroundColor = Color.White,
                    contentColor = Color.Black,
                    title = {
                        Text("Calculate Heart Attack Risk!")
                    }
                )
            },
            floatingActionButton = {
                Column {
                    FloatingActionButton(
                        onClick = {
                            coroutineScope.launch {
                                heartRateState.value = readAllData()
                                Log.i("heart rate", "heart Rate : $heartRateState.value")
                            }

                            //navController.navigate(Route.home.name)
                        },
                        modifier = Modifier.padding(bottom = 8.dp),
                        backgroundColor = Color.Black,
                        contentColor = Color.White,
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh",
                            tint = Color.White)
                    }

                    FloatingActionButton(
                        onClick = {
                            lifecycleScope.launch {
                                runCatching {
                                    //val json = Json.encodeToString(Data(heartRateState.value.toInt()))
                                    val response: HttpResponse =
                                        client.post("http://192.168.0.15:5000/rf") {
                                            contentType(ContentType.Application.Json)
                                            setBody(
                                                Data(
                                                    heartRate = heartRateState.value.toInt(),
                                                    age = age.toInt(),
                                                    cholesterol = cholesterol.toInt(),
                                                    diabetes = diabetes.toInt(),
                                                    familyHistory = familyHistory.toInt(),
                                                    smoking = smoking.toInt(),
                                                    obesity = obesity.toInt(),
                                                    alcoholConsumption = alcoholConsumption.toInt(),
                                                    exerciseHoursPerWeek = exerciseHoursPerWeek.toInt(),
                                                    diet = diet.toInt(),
                                                    previousHeartProblems = previousHeartProblems.toInt(),
                                                    medicationUse = medicationUse.toInt(),
                                                    stressLevel = stressLevel.toInt(),
                                                    sedentaryHoursPerDay = sedentaryHoursPerDay.toInt(),
                                                    bmi = bmi.toFloat(),
                                                    triglycerides = triglycerides.toInt(),
                                                    physicalActivityDaysPerWeek = physicalActivityDaysPerWeek.toInt(),
                                                    sleepHoursPerDay = sleepHoursPerDay.toInt(),
                                                    isMale = isMale.toInt(),
                                                    systolicPressure = systolicPressure.toInt(),
                                                    diastolicPressure = diastolicPressure.toInt()
                                                )
                                            )
                                        }

                                    if (!response.status.isSuccess()) {
                                        Log.i("ERROR - not success", response.status.description)
                                        return@launch
                                    }

                                    val x = response.body<String>()

                                    risk = x == "Heart Attack Risk Detected !!!"

                                    showRiskDialog = true

                                    Log.i("bilgilendirme", " $x")
                                }.onFailure {
                                    Log.e("ERROR", it.message.toString(), it)
                                }
                            }

                        },
                        backgroundColor = Color.Black,
                        contentColor = Color.White,
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.Send, contentDescription = "Send Data",
                            tint = Color.White)
                    }
                }
            }) { innerPadding ->

            Column(
                modifier = Modifier
                    .padding(innerPadding)
                    .fillMaxWidth()


            ) {
                val scrollState = rememberScrollState()
                var isLoggedIn by remember { mutableStateOf(false) }

                if (auth.currentUser != null) {
                    isLoggedIn = true
                }

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 16.dp, start = 16.dp, end = 16.dp)
                ) {
                    Text(
                        text = "Heart Rate",
                        fontSize = 16.sp,
                        color = Color.DarkGray,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )

                    Text(
                      text = "${heartRateState.value}",
                        style = MaterialTheme.typography.headlineLarge,
                        color = Color.Black,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(scrollState)
                    ) {
                        HealthInfoForm(
                            age = age,
                            onAgeChange = { age = it },
                            cholesterol = cholesterol,
                            onCholesterolChange = { cholesterol = it },
                            diabetes = diabetes,
                            onDiabetesChange = { diabetes = it },
                            familyHistory = familyHistory,
                            onFamilyHistoryChange = { familyHistory = it },
                            smoking = smoking,
                            onSmokingChange = { smoking = it },
                            obesity = obesity,
                            onObesityChange = { obesity = it },
                            alcoholConsumption = alcoholConsumption,
                            onAlcoholConsumptionChange = { alcoholConsumption = it },
                            exerciseHoursPerWeek = exerciseHoursPerWeek,
                            onExerciseHoursPerWeekChange = { exerciseHoursPerWeek = it },
                            diet = diet,
                            onDietChange = { diet = it },
                            previousHeartProblems = previousHeartProblems,
                            onPreviousHeartProblemsChange = { previousHeartProblems = it },
                            medicationUse = medicationUse,
                            onMedicationUseChange = { medicationUse = it },
                            stressLevel = stressLevel,
                            onStressLevelChange = { stressLevel = it },
                            sedentaryHoursPerDay = sedentaryHoursPerDay,
                            onSedentaryHoursPerDayChange = { sedentaryHoursPerDay = it },
                            bmi = bmi,
                            onBmiChange = { bmi = it },
                            triglycerides = triglycerides,
                            onTriglyceridesChange = { triglycerides = it },
                            physicalActivityDaysPerWeek = physicalActivityDaysPerWeek,
                            onPhysicalActivityDaysPerWeekChange = {
                                physicalActivityDaysPerWeek = it
                            },
                            sleepHoursPerDay = sleepHoursPerDay,
                            onSleepHoursPerDayChange = { sleepHoursPerDay = it },
                            isMale = isMale,
                            onIsMaleChange = { isMale = it },
                            systolicPressure = systolicPressure,
                            onSystolicPressureChange = { systolicPressure = it },
                            diastolicPressure = diastolicPressure,
                            onDiastolicPressureChange = { diastolicPressure = it },
                        )

                        Button(
                            onClick = {

                                Log.i("veriler ekleniyor", "...")
                                saveDataToFirebase(
                                    auth.currentUser!!.uid, HealthInfo(
                                        age = age,
                                        cholesterol = cholesterol,
                                        diabetes = diabetes,
                                        familyHistory = familyHistory,
                                        smoking = smoking,
                                        obesity = obesity,
                                        alcoholConsumption = alcoholConsumption,
                                        exerciseHoursPerWeek = exerciseHoursPerWeek,
                                        diet = diet,
                                        previousHeartProblems = previousHeartProblems,
                                        medicationUse = medicationUse,
                                        stressLevel = stressLevel,
                                        sedentaryHoursPerDay = sedentaryHoursPerDay,
                                        bmi = bmi,
                                        triglycerides = triglycerides,
                                        physicalActivityDaysPerWeek = physicalActivityDaysPerWeek,
                                        sleepHoursPerDay = sleepHoursPerDay,
                                        isMale = isMale,
                                        systolicPressure = systolicPressure,
                                        diastolicPressure = diastolicPressure
                                    )
                                )
                                Log.i("veriler eklendi", " . ")
                            },
                            modifier = Modifier.padding(top = 16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color.White,
                                contentColor = Color.Black
                            ),
                            border = BorderStroke(
                                width = 2.dp,
                                color = Color.DarkGray
                            )
                        ) {
                            Text("Save  Values")
                        }

                        Button(
                            onClick = {

                                coroutineScope.launch {
                                    val fetchedData = fetchDataFromFirebase(auth.currentUser!!.uid)

                                    age = fetchedData?.age ?: ""
                                    cholesterol = fetchedData?.cholesterol ?: ""
                                    diabetes = fetchedData?.diabetes ?: ""
                                    familyHistory = fetchedData?.familyHistory ?: ""
                                    smoking = fetchedData?.smoking ?: ""
                                    obesity = fetchedData?.obesity ?: ""
                                    alcoholConsumption = fetchedData?.alcoholConsumption ?: ""
                                    exerciseHoursPerWeek = fetchedData?.exerciseHoursPerWeek ?: ""
                                    diet = fetchedData?.diet ?: ""
                                    previousHeartProblems = fetchedData?.previousHeartProblems ?: ""
                                    medicationUse = fetchedData?.medicationUse ?: ""
                                    stressLevel = fetchedData?.stressLevel ?: ""
                                    sedentaryHoursPerDay = fetchedData?.sedentaryHoursPerDay ?: ""
                                    bmi = fetchedData?.bmi ?: ""
                                    triglycerides = fetchedData?.triglycerides ?: ""
                                    physicalActivityDaysPerWeek =
                                        fetchedData?.physicalActivityDaysPerWeek ?: ""
                                    sleepHoursPerDay = fetchedData?.sleepHoursPerDay ?: ""
                                    isMale = fetchedData?.isMale ?: ""
                                    systolicPressure = fetchedData?.systolicPressure ?: ""
                                    diastolicPressure = fetchedData?.diastolicPressure ?: ""

                                    loading = false


                                }
                            },
                            modifier = Modifier.padding(top = 16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color.White,
                                contentColor = Color.Black
                            ),
                            border = BorderStroke(
                                width = 2.dp,
                                color = Color.DarkGray
                            )
                        ) {
                            Text("Get  Values")
                        }

                    }
                }
            }

            if (showRiskDialog) {
                RiskDialog(
                    risk = risk,
                    onDismissRequest = { showRiskDialog = false }
                )
            }
        }
    }

    private fun saveDataToFirebase(userId: String, data: HealthInfo) {
        val db = FirebaseFirestore.getInstance()

        db.collection("users").document(userId).set(data)
            .addOnSuccessListener {
                Log.d("Firebase", "Data saved successfully")
            }
            .addOnFailureListener { e ->
                Log.w("Firebase", "Error adding document", e)
            }
    }

    private suspend fun fetchDataFromFirebase(userId: String): HealthInfo? {
        val db = FirebaseFirestore.getInstance()
        var healthInfo: HealthInfo? = null

        try {
            val snapshot = db.collection("users").document(userId).get().await()
            healthInfo = snapshot.toObject<HealthInfo>()
        } catch (e: Exception) {
            Log.e("Firebase", "Error fetching data", e)
        }

        return healthInfo
    }

    enum class Route {
        login, home, homeDetails

    }

    @Composable
    fun Navigation() {
        val navController = rememberNavController()
        NavHost(navController = navController, startDestination = Route.login.name) {
            composable(route = Route.login.name) {
                LoginScreen(
                    navController = navController
                )
            }
            composable(route = Route.home.name) {
                MainScreen(navController = navController)
            }
            composable(route = Route.home.name) {
                MainScreen(navController = navController)
            }
            composable(route = Route.homeDetails.name) {
                MainScreen(navController = navController)
            }

        }
    }


    private val client = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = true
                isLenient = true
            })
        }

        install(Logging) {

            level = LogLevel.BODY
        }
    }

    @Serializable
    data class Data(
        val heartRate: Int,
        val age: Int,
        val cholesterol: Int,
        val diabetes: Int,
        val familyHistory: Int,
        val smoking: Int,
        val obesity: Int,
        val alcoholConsumption: Int,
        val exerciseHoursPerWeek: Int,
        val diet: Int,
        val previousHeartProblems: Int,
        val medicationUse: Int,
        val stressLevel: Int,
        val sedentaryHoursPerDay: Int,
        val bmi: Float,
        val triglycerides: Int,
        val physicalActivityDaysPerWeek: Int,
        val sleepHoursPerDay: Int,
        val isMale: Int,
        val systolicPressure: Int,
        val diastolicPressure: Int
    )
}

@Composable
fun RiskDialog(
    risk: Boolean,
    onDismissRequest: () -> Unit
) {
    val riskText = if (risk) "Heart Attack Risk Detected !!!" else "No Risk of Heart Attack"
    val anim = if (risk) R.raw.risk_anim else R.raw.no_risk

    val composition by rememberLottieComposition(LottieCompositionSpec.RawRes(anim))

    Dialog(onDismissRequest = { onDismissRequest() }) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                LottieAnimation(
                    composition,
                    reverseOnRepeat = true,
                    iterations = LottieConstants.IterateForever,
                    modifier = Modifier.size(100.dp)
                )

                Text(
                    text = riskText,
                    modifier = Modifier
                        .fillMaxSize()
                        .wrapContentSize(Alignment.Center),
                    textAlign = TextAlign.Center,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 24.sp,
                )
            }
        }
    }
}
