
package com.example.hbeats

import android.content.Context
import android.os.RemoteException
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission

import androidx.health.connect.client.records.HeartRateRecord


import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import java.io.IOException
import java.time.Instant
import java.util.UUID

class HealthConnectManager(private val context: Context) {
    private val healthConnectClient by lazy { HealthConnectClient.getOrCreate(context) }

    val permissions = setOf(
        HealthPermission.getReadPermission(HeartRateRecord::class),
    )


    private var permissionsGranted = mutableStateOf(false)

    val requestPermissionActivityContract = PermissionController.createRequestPermissionResultContract()

    private var uiState: UiState by mutableStateOf(UiState.Uninitialized)

    suspend fun hasAllPermissions(): Boolean {
        return healthConnectClient.permissionController.getGrantedPermissions()
            .containsAll(permissions)
    }

    suspend fun readHeartRate(start: Instant, end: Instant): Any {
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


    suspend fun tryWithPermissionsCheck(block: suspend () -> Unit) {
        permissionsGranted.value = hasAllPermissions()
        uiState = try {
            if (permissionsGranted.value) {
                block()
            }
            UiState.Done
        } catch (remoteException: RemoteException) {
            UiState.Error(remoteException)
        } catch (securityException: SecurityException) {
            UiState.Error(securityException)
        } catch (ioException: IOException) {
            UiState.Error(ioException)
        } catch (illegalStateException: IllegalStateException) {
            UiState.Error(illegalStateException)
        }
    }

    sealed class UiState {
        object Uninitialized : UiState()
        object Done : UiState()
        data class Error(val exception: Throwable, val uuid: UUID = UUID.randomUUID()) : UiState()
    }
}
