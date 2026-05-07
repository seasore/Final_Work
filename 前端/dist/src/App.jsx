import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import Layout from './Layout'
import Dashboard from './pages/Dashboard'
import CorrelationAnalysis from './pages/analysis/CorrelationAnalysis'
import ManualPredict from './pages/predict/ManualPredict'
import ReportPredict from './pages/predict/ReportPredict'
import HolidayAnalysis from './pages/holiday/HolidayAnalysis'
import HolidayConfig from './pages/holiday/HolidayConfig'
import Eval96 from './pages/evaluate/Eval96'
import EvalHoliday from './pages/evaluate/EvalHoliday'
import EvalReport from './pages/evaluate/EvalReport'
import Assessment from './pages/Assessment'
import DataManage from './pages/DataManage'
import AdminUsers from './pages/admin/Users'
import AdminMessages from './pages/admin/Messages'
import AdminReports from './pages/admin/Reports'
import AdminLogs from './pages/admin/Logs'
import MemberReports from './pages/MemberReports'
import MemberMessages from './pages/MemberMessages'
import SettingsUsers from './pages/settings/Users'
import SettingsUnits from './pages/settings/Units'
import SettingsHoliday from './pages/settings/Holiday'
import SettingsModel from './pages/settings/Model'
import BigScreen from './pages/BigScreen'

function PrivateRoute({ children, adminOnly }) {
  const user = JSON.parse(localStorage.getItem('user') || 'null')
  if (!user) return <Navigate to="/login" replace />
  if (adminOnly && !['total_admin', 'group_admin'].includes(user.role)) {
    return <Navigate to="/" replace />
  }
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<PrivateRoute><Layout /></PrivateRoute>}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="analysis/correlation" element={<CorrelationAnalysis />} />
          <Route path="predict/manual" element={<ManualPredict />} />
          <Route path="predict/report" element={<ReportPredict />} />
          <Route path="holiday/analysis" element={<HolidayAnalysis />} />
          <Route path="holiday/config" element={<HolidayConfig />} />
          <Route path="evaluate/96" element={<Eval96 />} />
          <Route path="evaluate/holiday" element={<EvalHoliday />} />
          <Route path="evaluate/report" element={<EvalReport />} />
          <Route path="assessment" element={<Assessment />} />
          <Route path="data" element={<DataManage />} />
          <Route path="settings/users" element={<PrivateRoute adminOnly><SettingsUsers /></PrivateRoute>} />
          <Route path="settings/units" element={<PrivateRoute adminOnly><SettingsUnits /></PrivateRoute>} />
          <Route path="settings/messages" element={<PrivateRoute adminOnly><AdminMessages /></PrivateRoute>} />
          <Route path="settings/holiday" element={<PrivateRoute adminOnly><SettingsHoliday /></PrivateRoute>} />
          <Route path="settings/model" element={<PrivateRoute adminOnly><SettingsModel /></PrivateRoute>} />
          <Route path="reports" element={<MemberReports />} />
          <Route path="messages" element={<MemberMessages />} />
          <Route path="admin/users" element={<PrivateRoute adminOnly><AdminUsers /></PrivateRoute>} />
          <Route path="admin/messages" element={<PrivateRoute adminOnly><AdminMessages /></PrivateRoute>} />
          <Route path="admin/reports" element={<PrivateRoute adminOnly><AdminReports /></PrivateRoute>} />
          <Route path="admin/logs" element={<PrivateRoute adminOnly><AdminLogs /></PrivateRoute>} />
        </Route>
        <Route path="/bigscreen" element={<PrivateRoute><BigScreen /></PrivateRoute>} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
