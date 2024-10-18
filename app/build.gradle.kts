plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.jetbrainsKotlinAndroid)
    id("com.google.gms.google-services")


    kotlin("plugin.serialization") 
}

android {
    namespace = "com.example.hbeats"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.hbeats"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }


    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    viewBinding {
        enable = true

    }

    dependencies {

        implementation(libs.androidx.core.ktx)
        implementation(libs.androidx.lifecycle.runtime.ktx)
        implementation(libs.androidx.activity.compose)
        implementation(platform(libs.androidx.compose.bom))
        implementation(libs.androidx.ui)
        implementation(libs.androidx.ui.graphics)
        implementation(libs.androidx.ui.tooling.preview)
        implementation(libs.androidx.material3)
        testImplementation(libs.junit)
        androidTestImplementation(libs.androidx.junit)
        androidTestImplementation(libs.androidx.espresso.core)
        androidTestImplementation(platform(libs.androidx.compose.bom))
        androidTestImplementation(libs.androidx.ui.test.junit4)
        debugImplementation(libs.androidx.ui.tooling)
        debugImplementation(libs.androidx.ui.test.manifest)

        implementation(platform("com.google.firebase:firebase-bom:33.0.0"))
        implementation("com.google.firebase:firebase-auth")
        implementation("com.google.android.gms:play-services-auth:21.0.0")

        implementation("androidx.core:core-ktx:1.12.0")
        implementation("androidx.appcompat:appcompat:1.6.1")
        implementation("com.google.android.material:material:1.10.0")
        implementation("androidx.constraintlayout:constraintlayout:2.1.4")

        implementation("androidx.health.connect:connect-client:1.1.0-alpha06")

        implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2")
        implementation("androidx.navigation:navigation-compose:2.7.5")
        implementation("androidx.concurrent:concurrent-futures-ktx:1.1.0")

        // Icons
        implementation("androidx.compose.material:material-icons-extended:1.5.4")

        implementation("androidx.compose.ui:ui:1.5.4")
        implementation("androidx.compose.material:material:1.5.4")
        implementation("androidx.compose.ui:ui-tooling-preview:1.5.4")
        implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
        implementation("androidx.activity:activity-compose:1.8.1")

        debugImplementation("androidx.compose.ui:ui-tooling:1.5.4")

        coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4")

        implementation("androidx.credentials:credentials:1.3.0-alpha03")
        implementation("androidx.credentials:credentials-play-services-auth:1.3.0-alpha03")
        implementation("com.google.android.libraries.identity.googleid:googleid:1.1.0")

        // piacsso
        implementation ("com.squareup.picasso:picasso:2.71828")

        //splash screen
        implementation("com.airbnb.android:lottie-compose:6.4.0")


        //navigation

        implementation("androidx.navigation:navigation-compose:2.7.7")

        // ktor
        val ktorVersion = "2.3.5"

        implementation("io.ktor:ktor-client-core:$ktorVersion")
        implementation ("io.ktor:ktor-client-android:$ktorVersion")
        implementation ("io.ktor:ktor-client-logging:$ktorVersion")
        implementation ("io.ktor:ktor-client-content-negotiation:$ktorVersion")
        implementation ("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")


        implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")


    }
}
dependencies {
    implementation(libs.firebase.firestore.ktx)
}

